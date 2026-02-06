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
# This path must exist *inside the toolbox container*. In practice $HOME is mounted in.
RUN_LLAMA_IN_CONTAINER="${SIMON_LLAMA_RUN_LLAMA_IN_CONTAINER:-$HOME/run_llama.sh}"

MODELS_DIR="${SIMON_LLAMA_MODELS_DIR:-}"
HOST="${SIMON_LLAMA_HOST:-}"
PORT="${SIMON_LLAMA_PORT:-}"

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

# Important: pass args as an argv array (no bash -lc, no string concatenation).
exec env "${extra_env[@]}" toolbox run -c "$TOOLBOX_CONTAINER" "$RUN_LLAMA_IN_CONTAINER" "$cmd" "$@"

