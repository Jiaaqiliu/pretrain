#!/usr/bin/env bash
# Compact status snapshot for an in-flight distillation pipeline run.
#
# Usage:
#   monitor.sh <pipeline_name> [<workspace_root>]
#
# Defaults workspace_root to evolution_workdir/w4_baseline.
# Reports:
#   - tmux session liveness
#   - exit_code sentinel (if present)
#   - row counts of each stage's output JSONL
#   - last 12 lines of run.log
#   - last "progress" line from each stage so you can read pass-rate live
set -u

PIPELINE="${1:-}"
[ -z "$PIPELINE" ] && { echo "usage: $0 <pipeline_name> [<workspace_root>]" >&2; exit 2; }
WS_ROOT="${2:-/fsx/zzsamshi/a-evolve/evolution_workdir/w4_baseline}"
OUT_DIR="$WS_ROOT/artifacts/generation/$PIPELINE"
LOG_PATH="$OUT_DIR/run.log"
EXIT_PATH="$OUT_DIR/.exit_code"
SESSION="ne-distill-$PIPELINE"
MARKER="$WS_ROOT/.pending_jobs/distill-${PIPELINE}.json"

print_kv() { printf "  %-22s %s\n" "$1" "$2"; }

echo "=== pipeline: $PIPELINE ==="
print_kv "out_dir:" "$OUT_DIR"
print_kv "session:" "$SESSION"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  print_kv "tmux state:" "alive"
else
  print_kv "tmux state:" "GONE"
fi

if [ -f "$EXIT_PATH" ]; then
  print_kv "exit code:" "$(cat "$EXIT_PATH")"
else
  print_kv "exit code:" "(not yet written — run ongoing)"
fi

if [ -f "$MARKER" ]; then
  print_kv "marker:" "$MARKER (pending — run dw-pipeline-collect after exit)"
else
  print_kv "marker:" "(none — either not launched, or already harvested → done/)"
fi

echo
echo "=== stage row counts ==="
for f in stage1.jsonl stage2_teacher.jsonl stage3_self.jsonl stage4_audit.jsonl; do
  p="$OUT_DIR/$f"
  if [ -s "$p" ]; then
    n=$(wc -l < "$p")
    sz=$(du -h "$p" 2>/dev/null | cut -f1)
    printf "  %-24s %8d rows  (%s)\n" "$f" "$n" "$sz"
  elif [ -f "$p" ]; then
    printf "  %-24s %s\n" "$f" "(empty)"
  else
    printf "  %-24s %s\n" "$f" "—"
  fi
done

CURATED=$(ls "$OUT_DIR"/curated/*/[a-z]*_distilled.jsonl 2>/dev/null | head -1)
if [ -n "$CURATED" ]; then
  n=$(wc -l < "$CURATED")
  printf "  %-24s %8d rows  (%s)\n" "curated/<hash>/…" "$n" "$CURATED"
else
  printf "  %-24s %s\n" "curated/<hash>/…" "—"
fi

echo
echo "=== latest progress per stage ==="
for s in stage_1 stage_2 stage_3 stage_4 stage_5; do
  line=$(grep -E "(== ${s}_|${s}: rows_with_hit|${s}: scanned|${s}: pass=|^\\s*progress )" "$LOG_PATH" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    printf "  [%s] %s\n" "$s" "$line"
  fi
done

echo
echo "=== last 12 log lines ==="
tail -12 "$LOG_PATH" 2>/dev/null || echo "  (no log yet)"
