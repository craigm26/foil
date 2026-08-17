#!/usr/bin/env bash
# Install the local pre-push test gate. Hooks are not committed by git, so this
# is a one-time step per clone:  ./tools/install-hooks.sh
# There is deliberately no CI; the gate is local per the repo's deploy doctrine.
set -euo pipefail
cd "$(dirname "$0")/.."
cat > .git/hooks/pre-push <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
echo "pre-push: running the test suite (skip with --no-verify at your peril)"
./run.sh test
HOOK
chmod +x .git/hooks/pre-push
echo "installed .git/hooks/pre-push"
