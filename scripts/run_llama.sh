#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (override via env vars)
# -----------------------------
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1234}"

# Server mode:
# - "single": run llama-server with one explicit model (-m) (good for VL/mmproj selection)
# - "router": run llama-server in multi-model mode (no -m) with HTTP model management (/models, /models/unload)
SERVER_MODE="${SERVER_MODE:-single}" # single|router

# Router mode controls
MODELS_MAX="${MODELS_MAX:-1}"                 # number of models kept loaded (LRU)
NO_MODELS_AUTOLOAD="${NO_MODELS_AUTOLOAD:-off}" # on|off

CTX="${CTX:-131072}"
NGL="${NGL:-999}"
FA="${FA:-on}"

BATCH="${BATCH:-1024}"            # -b / --batch-size
UBATCH_SIZE="${UBATCH_SIZE:-256}" # -ub / --ubatch-size

THREADS="${THREADS:-$(nproc)}"
THREADS_BATCH="${THREADS_BATCH:-$THREADS}"

CACHE_K="${CACHE_K:-q8_0}"
CACHE_V="${CACHE_V:-q8_0}"

# Prompt cache controls (llama-server)
CACHE_PROMPT="${CACHE_PROMPT:-on}"          # on|off
CACHE_REUSE="${CACHE_REUSE:-256}"           # 0 disables reuse
SLOT_PROMPT_SIMILARITY="${SLOT_PROMPT_SIMILARITY:-0.10}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-}"        # set to enable disk persistence

PERF="${PERF:-off}" # on|off
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Runtime state (pid/log/lock)
RUNTIME_DIR="${RUNTIME_DIR:-$HOME/.local/state/llama-server}"
PID_FILE="$RUNTIME_DIR/llama-server.pid"
LOG_FILE="$RUNTIME_DIR/llama-server.log"
LOCK_FILE="$RUNTIME_DIR/llama-server.lock"
MODEL_FILE="$RUNTIME_DIR/llama-server.model"

# -----------------------------
# Helpers
# -----------------------------
die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

need_cmd llama-server
need_cmd find
need_cmd sort

mkdir -p "$RUNTIME_DIR"

[[ -d "$MODELS_DIR" ]] || die "MODELS_DIR not found: $MODELS_DIR"

json_escape() {
  local s="${1:-}"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

file_size_bytes() {
  local path="$1"
  if stat --version >/dev/null 2>&1; then
    stat -c '%s' "$path" 2>/dev/null || echo "0"
  else
    stat -f '%z' "$path" 2>/dev/null || echo "0"
  fi
}

file_mtime_unix() {
  local path="$1"
  if stat --version >/dev/null 2>&1; then
    stat -c '%Y' "$path" 2>/dev/null || echo "0"
  else
    stat -f '%m' "$path" 2>/dev/null || echo "0"
  fi
}

load_models() {
  # Exclude vision projector files from the "model" list.
  mapfile -t MODELS < <(find "$MODELS_DIR" -type f -name '*.gguf' ! -name '*mmproj*' | sort)
  ((${#MODELS[@]} > 0)) || die "No .gguf models found under: $MODELS_DIR"
}

list_mmproj_for_model() {
  local model_path="$1"
  local model_dir
  model_dir="$(dirname "$model_path")"
  find "$model_dir" -maxdepth 1 -type f -name '*mmproj*.gguf' 2>/dev/null | sort || true
}

resolve_first_shard_if_needed() {
  local model_path="$1"
  if [[ "$model_path" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
    local shard="${BASH_REMATCH[1]}"
    local total="${BASH_REMATCH[2]}"
    if [[ "$shard" != "00001" ]]; then
      local candidate="${model_path/-$shard-of-$total.gguf/-00001-of-$total.gguf}"
      if [[ -f "$candidate" ]]; then
        model_path="$candidate"
      fi
    fi
  fi
  printf "%s" "$model_path"
}

set_selected_model() {
  local model_path="$1"
  [[ -f "$model_path" ]] || die "Model not found: $model_path"
  MODEL_PATH="$(resolve_first_shard_if_needed "$model_path")"
  printf "%s" "$MODEL_PATH" > "$MODEL_FILE"
}

set_selected_mmproj() {
  local mmproj_path="$1"
  [[ -f "$mmproj_path" ]] || die "mmproj not found: $mmproj_path"
  MMPROJ_PATH="$mmproj_path"
}

pick_model() {
  load_models

  echo
  echo "Models in: $MODELS_DIR"
  echo "----------------------------------------"
  for i in "${!MODELS[@]}"; do
    idx=$((i+1))
    rel="${MODELS[$i]#"$MODELS_DIR"/}"
    printf "%3d) %s\n" "$idx" "$rel"
  done
  echo "----------------------------------------"
  echo

  read -r -p "Pick model number (1-${#MODELS[@]}): " PICK
  [[ "$PICK" =~ ^[0-9]+$ ]] || die "Not a number."
  ((PICK >= 1 && PICK <= ${#MODELS[@]})) || die "Out of range."

  set_selected_model "${MODELS[$((PICK-1))]}"

  # If this looks like a vision model, optionally pick mmproj from same dir
  MMPROJ_PATH=""
  if [[ "$MODEL_PATH" == *VL* || "$MODEL_PATH" == *vl* ]]; then
    model_dir="$(dirname "$MODEL_PATH")"
    mapfile -t MMPROJS < <(find "$model_dir" -maxdepth 1 -type f -name '*mmproj*.gguf' | sort)
    if ((${#MMPROJS[@]} > 0)); then
      echo
      read -r -p "Model looks vision-capable (VL). Load mmproj? (y/N): " WANT_MM
      if [[ "$WANT_MM" =~ ^[Yy]$ ]]; then
        echo
        echo "mmproj files in: $model_dir"
        echo "----------------------------------------"
        for i in "${!MMPROJS[@]}"; do
          idx=$((i+1))
          rel="${MMPROJS[$i]#"$MODELS_DIR"/}"
          printf "%3d) %s\n" "$idx" "$rel"
        done
        echo "----------------------------------------"
        echo
        read -r -p "Pick mmproj number (1-${#MMPROJS[@]}): " MPICK
        [[ "$MPICK" =~ ^[0-9]+$ ]] || die "Not a number."
        ((MPICK >= 1 && MPICK <= ${#MMPROJS[@]})) || die "Out of range."
        set_selected_mmproj "${MMPROJS[$((MPICK-1))]}"
        echo
        echo "Selected mmproj:"
        echo "  $MMPROJ_PATH"
        echo
      fi
    else
      echo
      echo "Note: model name contains 'VL' but no mmproj*.gguf found in:"
      echo "  $model_dir"
      echo
    fi
  fi

  echo
  echo "Selected model:"
  echo "  $MODEL_PATH"
  echo

  # MODEL_FILE already written by set_selected_model.
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

show_status() {
  if is_running; then
    local pid model
    pid="$(cat "$PID_FILE")"
    model="$(cat "$MODEL_FILE" 2>/dev/null || echo "unknown")"
    echo "RUNNING"
    echo "  PID:   $pid"
    echo "  MODE:  $SERVER_MODE"
    if [[ "$SERVER_MODE" == "router" ]]; then
      echo "  MODELS_DIR: $MODELS_DIR"
      echo "  MODELS_MAX: $MODELS_MAX"
    else
      echo "  MODEL: $model"
    fi
    echo "  HOST:  $HOST"
    echo "  PORT:  $PORT"
    echo "  LOG:   $LOG_FILE"
  else
    echo "STOPPED"
    [[ -f "$PID_FILE" ]] && echo "  (stale pidfile: $PID_FILE)"
  fi
}

show_status_json() {
  local running="false"
  local pid=""
  local model=""
  if is_running; then
    running="true"
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    model="$(cat "$MODEL_FILE" 2>/dev/null || true)"
  fi
  printf '{'
  printf '"running":%s,' "$running"
  printf '"pid":"%s",' "$(json_escape "$pid")"
  printf '"server_mode":"%s",' "$(json_escape "$SERVER_MODE")"
  printf '"model_path":"%s",' "$(json_escape "$model")"
  printf '"models_dir":"%s",' "$(json_escape "$MODELS_DIR")"
  printf '"models_max":%s,' "$(json_escape "$MODELS_MAX")"
  printf '"no_models_autoload":"%s",' "$(json_escape "$NO_MODELS_AUTOLOAD")"
  printf '"host":"%s",' "$(json_escape "$HOST")"
  printf '"port":%s,' "$(json_escape "$PORT")"
  printf '"log_file":"%s",' "$(json_escape "$LOG_FILE")"
  printf '"pid_file":"%s",' "$(json_escape "$PID_FILE")"
  printf '"model_file":"%s"' "$(json_escape "$MODEL_FILE")"
  printf '}\n'
}

list_models_human() {
  load_models
  echo "Models in: $MODELS_DIR"
  echo "----------------------------------------"
  for i in "${!MODELS[@]}"; do
    local idx=$((i+1))
    local rel="${MODELS[$i]#"$MODELS_DIR"/}"
    printf "%3d) %s\n" "$idx" "$rel"
  done
  echo "----------------------------------------"
}

list_models_json() {
  load_models
  printf '{'
  printf '"models_dir":"%s",' "$(json_escape "$MODELS_DIR")"
  printf '"count":%s,' "${#MODELS[@]}"
  printf '"models":['
  local first=1
  for i in "${!MODELS[@]}"; do
    local idx=$((i+1))
    local path="${MODELS[$i]}"
    local rel="${path#"$MODELS_DIR"/}"
    local size
    size="$(file_size_bytes "$path")"
    local mtime
    mtime="$(file_mtime_unix "$path")"
    mapfile -t mmprojs < <(list_mmproj_for_model "$path")
    [[ "$first" -eq 1 ]] || printf ','
    first=0
    printf '{'
    printf '"index":%s,' "$idx"
    printf '"path":"%s",' "$(json_escape "$path")"
    printf '"rel":"%s",' "$(json_escape "$rel")"
    printf '"size_bytes":%s,' "$size"
    printf '"mtime":%s,' "$mtime"
    printf '"mmprojs":['
    local mm_first=1
    for mp in "${mmprojs[@]}"; do
      [[ "$mm_first" -eq 1 ]] || printf ','
      mm_first=0
      printf '"%s"' "$(json_escape "$mp")"
    done
    printf ']'
    printf '}'
  done
  printf ']'
  printf '}\n'
}

build_cmd() {
  local perf_flag="--no-perf"
  [[ "$PERF" == "on" ]] && perf_flag="--perf"

  # shellcheck disable=SC2206
  CMD=(
    "$(command -v llama-server)"
    --no-mmap
    -ngl "$NGL"
    -fa "$FA"
    -c "$CTX"
    -b "$BATCH"
    --ubatch-size "$UBATCH_SIZE"
    --threads "$THREADS"
    --threads-batch "$THREADS_BATCH"
    --cache-type-k "$CACHE_K"
    --cache-type-v "$CACHE_V"
    "$perf_flag"
    --host "$HOST"
    --port "$PORT"
  )

  if [[ "$SERVER_MODE" == "router" ]]; then
    CMD+=(--models-dir "$MODELS_DIR" --models-max "$MODELS_MAX")
    if [[ "$NO_MODELS_AUTOLOAD" == "on" ]]; then
      CMD+=(--no-models-autoload)
    fi
  else
    CMD+=(-m "$MODEL_PATH")
    if [[ -n "${MMPROJ_PATH:-}" ]]; then
      CMD+=(--mmproj "$MMPROJ_PATH")
    fi
  fi

  # Prompt cache controls
  if [[ "$CACHE_PROMPT" == "on" ]]; then
    CMD+=(--cache-prompt)
  else
    CMD+=(--no-cache-prompt)
  fi

  if [[ -n "$CACHE_REUSE" && "$CACHE_REUSE" != "0" ]]; then
    CMD+=(--cache-reuse "$CACHE_REUSE")
  fi

  if [[ -n "$SLOT_PROMPT_SIMILARITY" ]]; then
    CMD+=(--slot-prompt-similarity "$SLOT_PROMPT_SIMILARITY")
  fi

  if [[ -n "$SLOT_SAVE_PATH" ]]; then
    mkdir -p "$SLOT_SAVE_PATH"
    CMD+=(--slot-save-path "$SLOT_SAVE_PATH")
  fi

  # Append EXTRA_ARGS (word-split intentionally for user-provided flags)
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=( $EXTRA_ARGS )
    CMD+=( "${EXTRA_ARR[@]}" )
  fi
}

start_server() {
  exec 9>"$LOCK_FILE"
  flock -n 9 || die "Another run-llama instance is controlling the server."

  if is_running; then
    echo "Already running."
    show_status
    return 0
  fi

  local model_arg="" index_arg="" mmproj_arg="" no_mmproj=0
  local server_mode_override=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --router)
        server_mode_override="router"
        shift
        ;;
      --single)
        server_mode_override="single"
        shift
        ;;
      --models-max)
        [[ $# -ge 2 ]] || die "--models-max requires a number"
        MODELS_MAX="$2"
        shift 2
        ;;
      --no-models-autoload)
        NO_MODELS_AUTOLOAD="on"
        shift
        ;;
      --model)
        [[ $# -ge 2 ]] || die "--model requires a path"
        model_arg="$2"
        shift 2
        ;;
      --index)
        [[ $# -ge 2 ]] || die "--index requires a number"
        index_arg="$2"
        shift 2
        ;;
      --mmproj)
        [[ $# -ge 2 ]] || die "--mmproj requires a path"
        mmproj_arg="$2"
        shift 2
        ;;
      --no-mmproj)
        no_mmproj=1
        shift
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done

  if [[ -n "$server_mode_override" ]]; then
    SERVER_MODE="$server_mode_override"
  fi

  if [[ "$SERVER_MODE" == "router" ]]; then
    if [[ -n "$model_arg" || -n "$index_arg" || -n "$mmproj_arg" || "$no_mmproj" -eq 1 ]]; then
      die "Router mode does not accept --model/--index/--mmproj/--no-mmproj. Use HTTP model management."
    fi
    MODEL_PATH=""
    MMPROJ_PATH=""
    printf "%s" "<router>" > "$MODEL_FILE"
  else
    MMPROJ_PATH=""
    if [[ -n "$model_arg" ]]; then
      set_selected_model "$model_arg"
    elif [[ -n "$index_arg" ]]; then
      [[ "$index_arg" =~ ^[0-9]+$ ]] || die "--index must be a number"
      load_models
      ((index_arg >= 1 && index_arg <= ${#MODELS[@]})) || die "--index out of range (1-${#MODELS[@]})"
      set_selected_model "${MODELS[$((index_arg-1))]}"
    else
      pick_model
    fi

    if [[ "$no_mmproj" -eq 1 ]]; then
      MMPROJ_PATH=""
    elif [[ -n "$mmproj_arg" ]]; then
      set_selected_mmproj "$mmproj_arg"
    fi
  fi

  build_cmd

  echo "Launching llama-server (background)…"
  echo "Log: $LOG_FILE"
  echo

  {
    echo "===== $(date -Is) START ====="
    echo "CMD: ${CMD[*]}"
  } >>"$LOG_FILE"

  # Start in background, detach from terminal
  nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  local pid=$!

  echo "$pid" > "$PID_FILE"

  # Quick health check: give it a moment, then verify PID alive
  sleep 0.3
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    die "llama-server exited immediately. Check log: $LOG_FILE"
  fi

  echo "Started."
  show_status
}

stop_server() {
  exec 9>"$LOCK_FILE"
  flock -n 9 || die "Another run-llama instance is controlling the server."

  if ! is_running; then
    echo "Not running."
    show_status
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping PID $pid …"

  kill "$pid" 2>/dev/null || true

  # Wait a bit then force kill if needed
  for _ in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Stopped."
      return 0
    fi
    sleep 0.1
  done

  echo "Graceful stop failed; sending SIGKILL."
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Stopped (forced)."
}

restart_server() {
  stop_server
  start_server "$@"
}

tail_logs() {
  [[ -f "$LOG_FILE" ]] || die "No log file yet: $LOG_FILE"
  tail -n 200 -f "$LOG_FILE"
}

menu() {
  echo
  echo "llama-server control (toolbox mode)"
  echo "----------------------------------------"
  echo "1) Start (pick model, run in background)"
  echo "2) Stop"
  echo "3) Restart (pick model)"
  echo "4) Status"
  echo "5) Logs (follow)"
  echo "6) Eject (free VRAM; stop server)"
  echo "0) Exit"
  echo "----------------------------------------"
  read -r -p "Choose: " CH
  case "$CH" in
    1) start_server ;;
    2) stop_server ;;
    3) restart_server ;;
    4) show_status ;;
    5) tail_logs ;;
    6) stop_server ;;
    0) exit 0 ;;
    *) echo "Nope." ;;
  esac
}

need_cmd flock

usage() {
  cat <<'EOF'
Usage:
  run_llama.sh                 # interactive menu
  run_llama.sh start [--router [--models-max N] [--no-models-autoload] | --single] [--model PATH | --index N] [--mmproj PATH | --no-mmproj]
  run_llama.sh stop            # stop server (eject/free VRAM)
  run_llama.sh eject           # alias for stop
  run_llama.sh restart [--router [--models-max N] [--no-models-autoload] | --single] [--model PATH | --index N] [--mmproj PATH | --no-mmproj]
  run_llama.sh status [--json]
  run_llama.sh list [--json]
  run_llama.sh logs

Router mode notes:
  - Start with: SERVER_MODE=router MODELS_MAX=1 run_llama.sh start --router
  - Model management happens over HTTP:
      GET  http://HOST:PORT/models
      POST http://HOST:PORT/models/unload {"model":"<id>"}
  - For multimodal models in --models-dir:
      Put model + mmproj into a subdirectory and name the projector file starting with "mmproj".
EOF
}

main() {
  if [[ $# -eq 0 ]]; then
    menu
    return 0
  fi

  local cmd="$1"
  shift || true
  case "$cmd" in
    start) start_server "$@" ;;
    stop) stop_server ;;
    eject|unload) stop_server ;;
    restart|switch) restart_server "$@" ;;
    status)
      if [[ "${1:-}" == "--json" ]]; then
        show_status_json
      else
        show_status
      fi
      ;;
    list)
      if [[ "${1:-}" == "--json" ]]; then
        list_models_json
      else
        list_models_human
      fi
      ;;
    logs) tail_logs ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
