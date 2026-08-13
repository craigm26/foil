#!/usr/bin/env bash
# Run any script in this repo with an Anthropic API key in the environment.
#
#   ./run.sh cli.py --n 50 run3 --episodes 12
#   ./run.sh pid2_run.py
#
# The key is taken from ANTHROPIC_API_KEY if set. Otherwise, if the 1Password
# CLI is available and FOIL_OP_ITEM points at an item, it is read at execution
# time and never written to disk or shell history.
#
#   export FOIL_OP_ITEM="op://<vault>/<item>/credential"
#
# These experiments make metered Messages API calls. A Claude Code subscription
# cannot serve them.

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -n "${FOIL_OP_ITEM:-}" ]; then
  [ -f "${OP_SERVICE_ACCOUNT_ENV:-$HOME/.config/op/service-account.env}" ] && {
    set -a; . "${OP_SERVICE_ACCOUNT_ENV:-$HOME/.config/op/service-account.env}"; set +a
  }
  # `op whoami` reports success even when reads fail, so test with a real read.
  if ! ANTHROPIC_API_KEY=$(op read "$FOIL_OP_ITEM" 2>/dev/null) || [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Could not read $FOIL_OP_ITEM" >&2; exit 2
  fi
  export ANTHROPIC_API_KEY
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Set ANTHROPIC_API_KEY, or FOIL_OP_ITEM to a 1Password secret reference." >&2
  exit 2
fi

exec python3 "$@"
