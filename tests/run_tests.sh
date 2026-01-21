#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/venv"

MODE="unit"
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit|-u)
      MODE="unit"
      shift
      ;;
    --integration|-i)
      MODE="integration"
      shift
      ;;
    --all|-a)
      MODE="all"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--unit|--integration|--all] [-- <pytest args>]"
      exit 0
      ;;
    --)
      shift
      PYTEST_ARGS+=("$@")
      break
      ;;
    *)
      PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -d "${VENV_DIR}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  else
    echo "venv not found at ${VENV_DIR}" >&2
    exit 1
  fi
fi

cd "${ROOT_DIR}"

run_unit() {
  pytest -q "${PYTEST_ARGS[@]}"
}

run_integration() {
  export SIMON_INTEGRATION=1
  unset SIMON_TEST_MODE || true
  pytest -q -o addopts= -m integration tests/integration "${PYTEST_ARGS[@]}"
}

case "${MODE}" in
  unit)
    run_unit
    ;;
  integration)
    run_integration
    ;;
  all)
    run_unit
    run_integration
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
