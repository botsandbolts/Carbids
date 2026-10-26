#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p .runtime
if ! python -c 'import streamlit, lightgbm' >/dev/null 2>&1; then
  python -m pip install -r requirements.txt
fi
if curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  echo 'CarBids Studio is already running on port 8501.'
else
  nohup python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > .runtime/studio.log 2>&1 < /dev/null &
  ready=false
  for attempt in {1..60}; do
    if curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 1
  done
  if [[ "$ready" != true ]]; then
    echo 'CarBids failed to start. Server log:' >&2
    tail -n 60 .runtime/studio.log >&2
    exit 1
  fi
fi
url='http://localhost:8501'
if [[ -n "${CODESPACE_NAME:-}" && -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]]; then
  url="https://${CODESPACE_NAME}-8501.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
fi
echo "CarBids Studio is ready: $url"
if [[ "${CARBIDS_OPEN_BROWSER:-1}" == 1 ]] && command -v code >/dev/null 2>&1; then
  code --open-external "$url" || true
fi
