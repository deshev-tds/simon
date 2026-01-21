#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/venv"

MODE="unit"
PYTEST_ARGS=()
PYTEST_BASE_OPTS=("-v")

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

if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
  for arg in "${PYTEST_ARGS[@]}"; do
    case "${arg}" in
      -q|-qq|-qqq|--quiet|-v|-vv|-vvv|--verbose)
        PYTEST_BASE_OPTS=()
        break
        ;;
    esac
  done
fi

if [[ -z "${VIRTUAL_ENV:-}" || "${VIRTUAL_ENV}" != "${VENV_DIR}" ]]; then
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
  local args=()
  if [[ ${#PYTEST_BASE_OPTS[@]} -gt 0 ]]; then
    args+=("${PYTEST_BASE_OPTS[@]}")
  fi
  if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    args+=("${PYTEST_ARGS[@]}")
  fi
  pytest "${args[@]}"
}

run_integration() {
  export SIMON_INTEGRATION=1
  unset SIMON_TEST_MODE || true
  local args=()
  if [[ ${#PYTEST_BASE_OPTS[@]} -gt 0 ]]; then
    args+=("${PYTEST_BASE_OPTS[@]}")
  fi
  local use_capture="yes"
  if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    for arg in "${PYTEST_ARGS[@]}"; do
      case "${arg}" in
        -s|--capture=no)
          use_capture="no"
          break
          ;;
      esac
    done
  fi
  if [[ "${use_capture}" == "yes" ]]; then
    args+=("-s")
  fi
  args+=("-o" "addopts=" "-m" "integration" "tests/integration")
  if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    args+=("${PYTEST_ARGS[@]}")
  fi
  pytest "${args[@]}"
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
