#!/usr/bin/env bash
# Sync this repo (go1_real) to the Go1 onboard PC at ~/go1_ws/src/go1_real.
#
#   ./scripts/sync_to_robot.sh            # dry run (shows what would transfer)
#   ./scripts/sync_to_robot.sh --go       # actually transfer
#
# Excludes .git (44M) and training artifacts the robot does not need.
set -euo pipefail

ROBOT="${ROBOT:-unitree@192.168.123.15}"
DEST="${DEST:-~/go1_ws/src/go1_real}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

DRY="--dry-run"
[[ "${1:-}" == "--go" ]] && DRY=""

ssh "$ROBOT" "mkdir -p $DEST"

rsync -avh --progress $DRY \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'events.out.tfevents.*' \
  --exclude 'outputs/' \
  "$SRC" "$ROBOT:$DEST/"

[[ -n "$DRY" ]] && echo && echo "DRY RUN only. Re-run with --go to transfer."
exit 0
