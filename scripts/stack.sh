#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
RUNTIME_DIR="$ROOT_DIR/.simon-run"
BACKEND_PID_FILE="$RUNTIME_DIR/backend.pid"
BACKEND_HTTP_PID_FILE="$RUNTIME_DIR/backend_http.pid"
FRONTEND_PID_FILE="$RUNTIME_DIR/frontend.pid"
BACKEND_LOG="$RUNTIME_DIR/backend.log"
BACKEND_HTTP_LOG="$RUNTIME_DIR/backend_http.log"
FRONTEND_LOG="$RUNTIME_DIR/frontend.log"
BACKEND_PORT=8000
BACKEND_HTTP_PORT=8001
FRONTEND_PORT=3000
FRONTEND_SCHEME="http"
TLS_AVAILABLE=0
if [[ -f "$ROOT_DIR/certs/key.pem" && -f "$ROOT_DIR/certs/cert.pem" ]]; then
  FRONTEND_SCHEME="https"
  TLS_AVAILABLE=1
fi

mkdir -p "$RUNTIME_DIR"

PYTHON_BIN="$BACKEND_DIR/venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

is_running() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  rm -f "$pidfile"
  return 1
}

print_status() {
  local label="$1"
  local pidfile="$2"
  local url="$3"
  if is_running "$pidfile"; then
    local pid
    pid="$(cat "$pidfile")"
    local uptime
    uptime="$(ps -p "$pid" -o etime= 2>/dev/null | xargs)"
    echo "$label: running (pid $pid${uptime:+, uptime $uptime}) @ $url"
  else
    echo "$label: stopped"
  fi
}

show_stats() {
  echo "Stats:"
  if [[ "$TLS_AVAILABLE" -eq 1 ]]; then
    print_status "Backend (TLS ${BACKEND_PORT})" "$BACKEND_PID_FILE" "https://localhost:${BACKEND_PORT}"
    print_status "Backend (HTTP ${BACKEND_HTTP_PORT})" "$BACKEND_HTTP_PID_FILE" "http://localhost:${BACKEND_HTTP_PORT}"
  else
    print_status "Backend (HTTP ${BACKEND_PORT})" "$BACKEND_PID_FILE" "http://localhost:${BACKEND_PORT}"
  fi
  print_status "Frontend (${FRONTEND_PORT})" "$FRONTEND_PID_FILE" "${FRONTEND_SCHEME}://localhost:${FRONTEND_PORT}"
}

kill_port_processes() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping process(es) on port ${port}: ${pids}"
    kill $pids 2>/dev/null || true
    sleep 0.5
    local remaining
    remaining="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$remaining" ]]; then
      echo "Force stopping process(es) on port ${port}: ${remaining}"
      kill -9 $remaining 2>/dev/null || true
    fi
  fi
}

start_backend() {
  if is_running "$BACKEND_PID_FILE" || is_running "$BACKEND_HTTP_PID_FILE"; then
    stop_backend
  fi
  kill_port_processes "$BACKEND_PORT"
  kill_port_processes "$BACKEND_HTTP_PORT"

  if [[ "$TLS_AVAILABLE" -eq 1 ]]; then
    echo "Starting backend (TLS on ${BACKEND_PORT})..."
    (cd "$ROOT_DIR" && nohup "$PYTHON_BIN" -m uvicorn backend.server:app --host 0.0.0.0 --port "$BACKEND_PORT" \
      --ssl-keyfile "$ROOT_DIR/certs/key.pem" --ssl-certfile "$ROOT_DIR/certs/cert.pem" \
      >"$BACKEND_LOG" 2>&1 & echo $! >"$BACKEND_PID_FILE")
    echo "Starting backend (HTTP on ${BACKEND_HTTP_PORT})..."
    (cd "$ROOT_DIR" && nohup "$PYTHON_BIN" -m uvicorn backend.server:app --host 0.0.0.0 --port "$BACKEND_HTTP_PORT" \
      >"$BACKEND_HTTP_LOG" 2>&1 & echo $! >"$BACKEND_HTTP_PID_FILE")
  else
    echo "Starting backend (HTTP on ${BACKEND_PORT})..."
    (cd "$ROOT_DIR" && nohup "$PYTHON_BIN" -m uvicorn backend.server:app --host 0.0.0.0 --port "$BACKEND_PORT" \
      >"$BACKEND_LOG" 2>&1 & echo $! >"$BACKEND_PID_FILE")
  fi

  sleep 1
  if [[ "$TLS_AVAILABLE" -eq 1 ]]; then
    if is_running "$BACKEND_PID_FILE" && is_running "$BACKEND_HTTP_PID_FILE"; then
      echo "Backend up."
      return 0
    fi
  else
    if is_running "$BACKEND_PID_FILE"; then
      echo "Backend up."
      return 0
    fi
  fi
  echo "Backend failed to start. See $BACKEND_LOG"
  return 1
}

start_frontend() {
  if is_running "$FRONTEND_PID_FILE"; then
    stop_frontend
  fi
  kill_port_processes "$FRONTEND_PORT"
  echo "Starting frontend on ${FRONTEND_PORT}..."
  (cd "$FRONTEND_DIR" && nohup npm run dev -- --host --port "$FRONTEND_PORT" --strictPort >"$FRONTEND_LOG" 2>&1 & echo $! >"$FRONTEND_PID_FILE")
  sleep 1
  if is_running "$FRONTEND_PID_FILE"; then
    echo "Frontend up."
    return 0
  fi
  echo "Frontend failed to start. See $FRONTEND_LOG"
  return 1
}

stop_backend() {
  if is_running "$BACKEND_PID_FILE"; then
    local pid
    pid="$(cat "$BACKEND_PID_FILE")"
    echo "Stopping backend (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  if is_running "$BACKEND_HTTP_PID_FILE"; then
    local pid
    pid="$(cat "$BACKEND_HTTP_PID_FILE")"
    echo "Stopping backend (HTTP pid $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -f "$BACKEND_PID_FILE" "$BACKEND_HTTP_PID_FILE"
}

stop_frontend() {
  if is_running "$FRONTEND_PID_FILE"; then
    local pid
    pid="$(cat "$FRONTEND_PID_FILE")"
    echo "Stopping frontend (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  rm -f "$FRONTEND_PID_FILE"
}

start_all() {
  local failure=0
  start_backend || failure=1
  start_frontend || failure=1
  return $failure
}

stop_all() {
  stop_frontend
  stop_backend
}

restart_all() {
  stop_all
  start_all
}

while true; do
  echo
  echo "Choose an action: 1) start 2) stop 3) restart 4) exit"
  read -rp "> " choice
  case "$choice" in
    1) start_all; show_stats ;;
    2) stop_all; show_stats ;;
    3) restart_all; show_stats ;;
    4) show_stats; exit 0 ;;
    *) echo "Please enter 1, 2, 3, or 4." ;;
  esac
done
