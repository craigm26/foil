#!/usr/bin/env bash
# Run any cli.py subcommand with the API key pulled from 1Password at
# execution time. The key is never written to disk, never echoed, and never
# enters shell history or a session transcript.
#
#   ./foil.sh plan
#   ./foil.sh --n 50 run2
#   ./foil.sh --n 200 run
#
# Setup is documented in run-nulls.sh.

set -euo pipefail
cd "$(dirname "$0")"

ITEM="FOIL Anthropic API key"
VAULT="Civqo"

# `op whoami` reports success even when reads fail, so the only honest check
# is an actual read.
set -a; . "$HOME/.config/op/service-account.env"; set +a

if ! KEY=$(op read "op://${VAULT}/${ITEM}/credential" 2>/dev/null) || [ -z "$KEY" ]; then
  echo "Could not read op://${VAULT}/${ITEM}/credential" >&2
  exit 2
fi

ANTHROPIC_API_KEY="$KEY" exec python3 pid_run.py "$@"
