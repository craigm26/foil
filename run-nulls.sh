#!/usr/bin/env bash
# Run the Phase 0 nulls with the API key pulled from 1Password at execution
# time. The key is never written to disk, never echoed, and never enters a
# shell history or a session transcript.
#
# Setup (once):
#   1. Create a key at console.anthropic.com scoped to this project.
#   2. Store it:
#        op item create --category="API Credential" --vault=Civqo \
#          --title="FOIL Anthropic API key" credential="sk-ant-..."
#      (or paste it into the 1Password UI as an API Credential item with that
#      exact title; the field must be `credential`)
#
# Usage:
#   ./run-nulls.sh            # n=200, claude-sonnet-5
#   ./run-nulls.sh --n 100    # any cli.py flags pass through

set -euo pipefail
cd "$(dirname "$0")"

ITEM="FOIL Anthropic API key"
VAULT="Civqo"

# Service-account token. `op whoami` reports success even when reads fail, so
# the only honest check is an actual read (below).
set -a; . "$HOME/.config/op/service-account.env"; set +a

if ! KEY=$(op read "op://${VAULT}/${ITEM}/credential" 2>/dev/null); then
  echo "Could not read op://${VAULT}/${ITEM}/credential" >&2
  echo "Create the item first -- see the header of this script." >&2
  exit 2
fi

if [ -z "$KEY" ]; then
  echo "Item read but credential field is empty." >&2
  exit 2
fi

# Default to the pre-registered n=200 (§7.3 amendment) unless overridden.
if [ "$#" -eq 0 ]; then
  set -- --n 200
fi

# Scoped to this process only. Not exported to the parent shell.
ANTHROPIC_API_KEY="$KEY" exec python3 cli.py "$@" run
