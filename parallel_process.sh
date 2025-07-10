#!/usr/bin/env bash
# parallel_process.sh — send the SAME PDF N times in parallel + stats
# ------------------------------------------------------------------
# Examples
#   ./parallel_process.sh                # 10 jobs, file = input.pdf
#   ./parallel_process.sh 25             # 25 jobs, file = input.pdf
#   ./parallel_process.sh 25 mydoc.pdf   # 25 jobs, file = mydoc.pdf

###############################################################################
ENDPOINT="http://pdf-an-Appli-SHny1a3n9zIU-1027984804.us-east-2.elb.amazonaws.com/visualize"
JOBS="${1:-10}"          # how many parallel requests
PDF="${2:-input.pdf}"    # file to POST
###############################################################################

set -euo pipefail

[[ -f "$PDF" ]] || { echo "❌  File '$PDF' not found."; exit 1; }

echo "→ Launching $JOBS parallel requests"
echo "  Source file : $PDF"
echo "  Output files: output_<idx>.pdf"

# temp file to collect per‑job results
TMP_STAT=$(mktemp)          # cleaned up automatically on most macOS/Linux
trap 'rm -f "$TMP_STAT"' EXIT

export ENDPOINT PDF TMP_STAT

start_ts=$(date +%s)

# ----------------------------------------------------------------------------- 
seq 1 "$JOBS" | \
xargs -I{} -P"$JOBS" sh -c '
  idx="$1"
  if curl -sS -X POST \
           -F "file=@${PDF}" \
           -F "fast=false" \
           -F "extraction_format=" \
           "${ENDPOINT}" \
           > "output_${idx}.pdf"; then
       echo "ok"   >> "$TMP_STAT"
       echo "✔︎  job ${idx} ok"
  else
       echo "fail" >> "$TMP_STAT"
       echo "⚠︎  job ${idx} failed"
  fi
' _ {}

# ----------------------------------------------------------------------------- 
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

successes=$(grep -c '^ok$'   "$TMP_STAT" || true)
fails=$(grep -c '^fail$' "$TMP_STAT" || true)
rate=$(awk "BEGIN{ printf \"%.1f\", ($successes/$JOBS)*100 }")

echo "------------------------------------------------------------------"
echo "✅  All $JOBS requests finished in ${elapsed}s"
echo "    Success : $successes"
echo "    Failed  : $fails"
echo "    Rate    : ${rate}%"

