#!/usr/bin/env bash
set -euo pipefail
mkdir -p .runtime
if curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  echo 'CarBids Studio is already running on port 8501.'
else
  nohup python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > .runtime/studio.log 2>&1 < /dev/null &
  echo 'CarBids Studio started. Open forwarded port 8501.'
fi
