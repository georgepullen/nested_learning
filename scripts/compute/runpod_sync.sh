#!/usr/bin/env bash
# Push/pull repository artifacts to/from a RunPod instance using rsync over SSH.
set -euo pipefail

MODE=${1:-}
TARGET=${2:-}
PORT=${3:-22}
REMOTE_DIR=${4:-/workspace/nested_learning}
LOCAL_DIR=${5:-$(pwd)}
RUNPOD_SSH_KEY=${RUNPOD_SSH_KEY:-}

if [[ -z "${MODE}" || -z "${TARGET}" ]]; then
  echo "Usage: $0 <push|pull> <user@host> [port] [remote_dir] [local_dir]"
  exit 1
fi

SSH_ARGS=(-p "${PORT}")
if [[ -n "${RUNPOD_SSH_KEY}" ]]; then
  SSH_ARGS+=(-i "${RUNPOD_SSH_KEY}" -o IdentitiesOnly=yes)
fi

if [[ "${MODE}" == "push" ]]; then
  _local_has_rsync=0
  _remote_has_rsync=0
  command -v rsync >/dev/null 2>&1 && _local_has_rsync=1
  if ssh "${SSH_ARGS[@]}" "${TARGET}" "command -v rsync >/dev/null 2>&1"; then
    _remote_has_rsync=1
  fi

  if [[ "${_local_has_rsync}" == "1" && "${_remote_has_rsync}" == "1" ]]; then
    rsync -az --delete \
      --exclude '.git' \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '.mypy_cache' \
      --exclude '.pytest_cache' \
      -e "ssh ${SSH_ARGS[*]}" \
      "${LOCAL_DIR}/" "${TARGET}:${REMOTE_DIR}/"
  else
    echo "[runpod-sync] rsync unavailable (local=${_local_has_rsync}, remote=${_remote_has_rsync}), falling back to tar-over-ssh push"
    tar \
      --exclude '.git' \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '.mypy_cache' \
      --exclude '.pytest_cache' \
      -C "${LOCAL_DIR}" -cf - . \
      | ssh "${SSH_ARGS[@]}" "${TARGET}" "mkdir -p \"${REMOTE_DIR}\" && tar -xf - -C \"${REMOTE_DIR}\""
  fi
  echo "[runpod-sync] pushed ${LOCAL_DIR} -> ${TARGET}:${REMOTE_DIR}"
elif [[ "${MODE}" == "pull" ]]; then
  mkdir -p "${LOCAL_DIR}/artifacts" "${LOCAL_DIR}/logs" "${LOCAL_DIR}/eval" "${LOCAL_DIR}/reports"
  _local_has_rsync=0
  _remote_has_rsync=0
  command -v rsync >/dev/null 2>&1 && _local_has_rsync=1
  if ssh "${SSH_ARGS[@]}" "${TARGET}" "command -v rsync >/dev/null 2>&1"; then
    _remote_has_rsync=1
  fi

  if [[ "${_local_has_rsync}" == "1" && "${_remote_has_rsync}" == "1" ]]; then
    rsync -az -e "ssh ${SSH_ARGS[*]}" "${TARGET}:${REMOTE_DIR}/artifacts/" "${LOCAL_DIR}/artifacts/" || true
    rsync -az -e "ssh ${SSH_ARGS[*]}" "${TARGET}:${REMOTE_DIR}/logs/" "${LOCAL_DIR}/logs/" || true
    rsync -az -e "ssh ${SSH_ARGS[*]}" "${TARGET}:${REMOTE_DIR}/eval/" "${LOCAL_DIR}/eval/" || true
    rsync -az -e "ssh ${SSH_ARGS[*]}" "${TARGET}:${REMOTE_DIR}/reports/" "${LOCAL_DIR}/reports/" || true
  else
    echo "[runpod-sync] rsync unavailable (local=${_local_has_rsync}, remote=${_remote_has_rsync}), falling back to tar-over-ssh pull"
    for _dir in artifacts logs eval reports; do
      if ssh "${SSH_ARGS[@]}" "${TARGET}" "test -d \"${REMOTE_DIR}/${_dir}\""; then
        ssh "${SSH_ARGS[@]}" "${TARGET}" "tar -cf - -C \"${REMOTE_DIR}\" \"${_dir}\"" \
          | tar -xf - -C "${LOCAL_DIR}"
      fi
    done
  fi
  echo "[runpod-sync] pulled artifacts/logs/eval/reports into ${LOCAL_DIR}"
else
  echo "[runpod-sync] unknown mode: ${MODE}"
  echo "Usage: $0 <push|pull> <user@host> [port] [remote_dir] [local_dir]"
  exit 1
fi
