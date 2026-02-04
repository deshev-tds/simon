#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/simon}"
APP_USER="${APP_USER:-simon}"
REPO_URL="${REPO_URL:-https://github.com/deshev-tds/simon.git}"
BRANCH="${BRANCH:-main}"
BOOTSTRAP="${BOOTSTRAP:-0}"
ALLOW_RESET="${ALLOW_RESET:-0}"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    exec sudo -E bash "$0" "$@"
  fi
}

run_as_app() {
  sudo -u "${APP_USER}" -H bash -lc "$*"
}

ensure_repo_clean() {
  if [[ ! -d "${APP_DIR}/.git" ]]; then
    echo "ERROR: ${APP_DIR} is not a git repo."
    echo "To initialize it from ${REPO_URL}, run:"
    echo "  BOOTSTRAP=1 ALLOW_RESET=1 ${0}"
    exit 1
  fi
  local status
  status="$(run_as_app "cd '${APP_DIR}' && git status --porcelain")"
  if [[ -n "${status}" ]]; then
    echo "ERROR: repo has uncommitted changes:"
    echo "${status}"
    exit 1
  fi
}

bootstrap_repo() {
  if [[ "${BOOTSTRAP}" -ne 1 ]]; then
    return 0
  fi
  if [[ "${ALLOW_RESET}" -ne 1 ]]; then
    echo "ERROR: BOOTSTRAP=1 requires ALLOW_RESET=1 (this will overwrite tracked files)."
    exit 1
  fi
  run_as_app "cd '${APP_DIR}' && git init"
  run_as_app "cd '${APP_DIR}' && git remote remove origin >/dev/null 2>&1 || true"
  run_as_app "cd '${APP_DIR}' && git remote add origin '${REPO_URL}'"
  run_as_app "cd '${APP_DIR}' && git fetch --prune origin"
  run_as_app "cd '${APP_DIR}' && git reset --hard 'origin/${BRANCH}'"
  run_as_app "cd '${APP_DIR}' && git submodule update --init --recursive"
}

pull_latest() {
  run_as_app "cd '${APP_DIR}' && git fetch --all --prune"
  run_as_app "cd '${APP_DIR}' && git pull --ff-only origin '${BRANCH}'"
}

install_backend_deps() {
  if [[ -x "${APP_DIR}/backend/venv/bin/pip" ]]; then
    run_as_app "cd '${APP_DIR}' && '${APP_DIR}/backend/venv/bin/pip' install -r backend/requirements.txt"
    # These are required at runtime but not pinned in requirements.txt
    run_as_app "cd '${APP_DIR}' && '${APP_DIR}/backend/venv/bin/pip' install fastapi 'uvicorn[standard]' chromadb"
  else
    echo "WARN: venv missing at ${APP_DIR}/backend/venv. Skipping backend deps."
  fi
}

install_frontend_deps() {
  if command -v npm >/dev/null 2>&1; then
    run_as_app "cd '${APP_DIR}/frontend' && npm install"
  else
    echo "WARN: npm not found. Skipping frontend deps."
  fi
}

restart_services() {
  systemctl restart simon-backend
  systemctl restart simon-frontend
}

main() {
  require_root "$@"
  bootstrap_repo
  ensure_repo_clean
  pull_latest
  install_backend_deps
  install_frontend_deps
  restart_services
  systemctl status --no-pager simon-backend simon-frontend || true
}

main "$@"
