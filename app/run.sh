#!/usr/bin/env bash
# Start the Scrubline console on http://127.0.0.1:8787
# Secrets stay outside the repo:
#   ~/.config/river/env   RIVER_API_KEY=...                         (Qwen via River)
#   ~/.config/gbrain/env  GBRAIN_MCP_URL=https://gbrain.io/mcp       (GBrain memory; optional)
#                         GBRAIN_TOKEN=...
cd "$(dirname "$0")/.."
set -a
[ -f ~/.config/river/env ] && . ~/.config/river/env
[ -f ~/.config/gbrain/env ] && . ~/.config/gbrain/env
set +a
exec .venv/bin/python app/server.py
