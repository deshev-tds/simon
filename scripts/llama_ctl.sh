#!/usr/bin/env bash
set -euo pipefail

# Wrapper used by Simon backend to control llama.cpp running inside a toolbox container.
#
# Typical call path:
#   (simon user) backend/server.py -> sudo -n -u deshev /opt/simon/scripts/llama_ctl.sh ...
#   (deshev user) llama_ctl.sh -> toolbox run -c llama-rocm-7.2 /home/deshev/run_llama.sh ...
#
# Security note:
# - This script must NOT eval user-provided input.
# - If you allow NOPASSWD sudo for it, keep the sudoers rule scoped to this file.

die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

need_cmd toolbox

TOOLBOX_CONTAINER="${SIMON_LLAMA_TOOLBOX_CONTAINER:-llama-rocm-7.2}"
RUN_LLAMA_IN_CONTAINER="${SIMON_LLAMA_RUN_LLAMA_IN_CONTAINER:-$HOME/run_llama.sh}"

MODELS_DIR="${SIMON_LLAMA_MODELS_DIR:-}"
HOST="${SIMON_LLAMA_HOST:-}"
PORT="${SIMON_LLAMA_PORT:-}"

# When called via sudo, rootless podman/toolbox often need these set.
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR:-}" ]]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "${XDG_RUNTIME_DIR}/bus" ]]; then
  export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
fi

# Avoid toolbox trying (and failing) to chdir into paths not mounted in the container (e.g. /opt/simon).
cd "${SIMON_LLAMA_HOST_CWD:-$HOME}" || die "Failed to cd to host cwd"

extra_env=()
[[ -n "$MODELS_DIR" ]] && extra_env+=("MODELS_DIR=$MODELS_DIR")
[[ -n "$HOST" ]] && extra_env+=("HOST=$HOST")
[[ -n "$PORT" ]] && extra_env+=("PORT=$PORT")

if [[ $# -eq 0 ]]; then
  die "No command. Try: list --json | status --json | restart --model PATH | eject"
fi

cmd="$1"
shift || true

case "$cmd" in
  list|status|start|stop|restart|switch|eject|unload|logs|help|-h|--help)
    ;;
  *)
    die "Unknown command: $cmd"
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host_run_llama="$script_dir/run_llama.sh"
if [[ ! -f "$host_run_llama" ]]; then
  die "Host run_llama.sh not found at: $host_run_llama"
fi

# Prefer an in-container script path if the user explicitly provides one,
# but fall back to piping the host script into the container so we don't
# depend on /home being mounted or symlinks resolving inside the container.
if [[ -n "${SIMON_LLAMA_RUN_LLAMA_IN_CONTAINER:-}" ]]; then
  exec env "${extra_env[@]}" toolbox run -c "$TOOLBOX_CONTAINER" "$RUN_LLAMA_IN_CONTAINER" "$cmd" "$@"
fi

# Important: pass args as an argv array (no string concatenation).
exec env "${extra_env[@]}" toolbox run -c "$TOOLBOX_CONTAINER" bash -s -- "$cmd" "$@" <"$host_run_llama"
