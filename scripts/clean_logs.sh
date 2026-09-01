#!/bin/bash
# Delete run logs that never reached main.py's "mean," summary, i.e. crashed attempts.
# Skips any row that still has a job running. Usage: clean_logs.sh <OUTROOT>
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUTROOT=${1:?usage: clean_logs.sh <OUTROOT>}

for d in "$OUTROOT"/table1/*/s*/; do
  [ -d "$d" ] || continue
  row=$(basename "$(dirname "$d")"); seed=$(basename "$d")
  if sessions | grep -q "_${row}_${seed}$"; then echo "skip (running): $row/$seed"; continue; fi
  for f in "$d"/*.log; do
    [ -f "$f" ] || continue
    grep -q "^mean," "$f" 2>/dev/null || { rm -f "$f"; echo "removed failed: ${f#"$OUTROOT"/}"; }
  done
done
