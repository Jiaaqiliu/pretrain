#!/usr/bin/env bash
# Status snapshot for distillation pipeline runs.
#
# Usage:
#   monitor.sh                       # all in-flight pipelines (enumerate
#                                      pending markers under .pending_jobs/)
#   monitor.sh <pipeline_name>       # one pipeline by name
#   monitor.sh --workspace <root>    # override workspace root (defaults
#                                      to evolution_workdir/w4_baseline)
#   monitor.sh --compact             # one-line-per-pipeline + 1-line
#                                      progress summary; designed for
#                                      `watch -n 10` so the screen never
#                                      scrolls past one viewport
#
# Default (verbose) reports per pipeline:
#   - tmux session liveness, exit_code, marker
#   - per-stage row counts
#   - latest progress line per stage
#   - last 8 lines of run.log
set -u

WS_ROOT_DEFAULT=/fsx/zzsamshi/a-evolve/evolution_workdir/w4_baseline
WS_ROOT="$WS_ROOT_DEFAULT"
PIPELINE=""
COMPACT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) WS_ROOT="$2"; shift 2 ;;
    --all)       PIPELINE=""; shift ;;
    --compact)   COMPACT=1; shift ;;
    -h|--help)
      sed -n '2,17p' "$0"; exit 0 ;;
    *)           PIPELINE="$1"; shift ;;
  esac
done

print_kv() { printf "  %-22s %s\n" "$1" "$2"; }

# Pull the most recent tqdm frame from a log. tqdm writes
# "stage_X:  NN%|██▎     | A/B [elapsed<eta, rate, postfix]" segments
# separated by carriage returns. We split on \r and keep the last one.
latest_tqdm_frame() {
  local log="$1"
  [ -f "$log" ] || { echo ""; return; }
  # tail to a bounded byte range so this is cheap even on long runs.
  tail -c 8000 "$log" | awk -v RS='\r' 'END { print }' \
    | grep -oE 'stage_[0-9]:[^[:cntrl:]]*' | tail -1 \
    | sed 's/[[:space:]]\+$//'
}

# Render the snapshot for one pipeline. Caller passes pipeline_name; we
# derive everything else from WS_ROOT.
render_one() {
  local name="$1"
  local out_dir="$WS_ROOT/artifacts/generation/$name"
  local log_path="$out_dir/run.log"
  local exit_path="$out_dir/.exit_code"
  local session="ne-distill-$name"
  local marker="$WS_ROOT/.pending_jobs/distill-${name}.json"

  echo "=== pipeline: $name ==="
  print_kv "out_dir:" "$out_dir"
  print_kv "session:" "$session"
  if tmux has-session -t "$session" 2>/dev/null; then
    print_kv "tmux state:" "alive (attach: tmux attach -t $session)"
  else
    print_kv "tmux state:" "GONE"
  fi

  if [ -f "$exit_path" ]; then
    print_kv "exit code:" "$(cat "$exit_path")"
  else
    print_kv "exit code:" "(not yet written — run ongoing)"
  fi

  if [ -f "$marker" ]; then
    print_kv "marker:" "$marker"
    print_kv "" "(pending — run dw-pipeline-collect after exit)"
  else
    if [ -f "$WS_ROOT/.pending_jobs/done/distill-${name}.json" ]; then
      print_kv "marker:" "(harvested → done/)"
    else
      print_kv "marker:" "(none — not launched)"
    fi
  fi

  echo
  echo "  --- stage row counts ---"
  local f p n sz
  for f in stage1.jsonl stage2_teacher.jsonl stage3_self.jsonl stage4_audit.jsonl; do
    p="$out_dir/$f"
    if [ -s "$p" ]; then
      n=$(wc -l < "$p")
      sz=$(du -h "$p" 2>/dev/null | cut -f1)
      printf "    %-24s %8d rows  (%s)\n" "$f" "$n" "$sz"
    elif [ -f "$p" ]; then
      printf "    %-24s %s\n" "$f" "(empty)"
    else
      printf "    %-24s %s\n" "$f" "—"
    fi
  done

  local curated
  curated=$(ls "$out_dir"/curated/*/[a-z]*_distilled.jsonl 2>/dev/null | head -1)
  if [ -n "$curated" ]; then
    n=$(wc -l < "$curated")
    printf "    %-24s %8d rows  (%s)\n" "curated/<hash>/…" "$n" "$curated"
  else
    printf "    %-24s %s\n" "curated/<hash>/…" "—"
  fi

  echo
  echo "  --- latest progress per stage ---"
  local s line
  for s in stage_1 stage_2 stage_3 stage_4 stage_5; do
    line=$(grep -E "(== ${s}_|${s}: rows_with_hit|${s}: scanned|${s}: pass=|^\\s*progress )" "$log_path" 2>/dev/null | tail -1)
    [ -n "$line" ] && printf "    [%s] %s\n" "$s" "$line"
  done

  echo
  echo "  --- last 8 log lines ---"
  if [ -f "$log_path" ]; then
    tail -8 "$log_path" | sed 's/^/    /'
  else
    echo "    (no log yet)"
  fi
}

# Compact: one row per pipeline, never more than ~3 lines per pipeline.
# Designed for `watch -n 10 monitor.sh --compact` so the screen never scrolls.
render_one_compact() {
  local name="$1"
  local out_dir="$WS_ROOT/artifacts/generation/$name"
  local log_path="$out_dir/run.log"
  local exit_path="$out_dir/.exit_code"
  local session="ne-distill-$name"
  local marker="$WS_ROOT/.pending_jobs/distill-${name}.json"

  # Status badge
  local status
  if tmux has-session -t "$session" 2>/dev/null; then
    status="RUN "
  elif [ -f "$exit_path" ]; then
    local rc
    rc=$(grep -o 'EXIT_CODE=[0-9-]*' "$exit_path" | head -1 | cut -d= -f2)
    if [ "${rc:-?}" = "0" ]; then status="DONE"; else status="FAIL"; fi
  else
    status="????"
  fi

  # Marker badge
  local mark
  if   [ -f "$marker" ];                                                  then mark="pending"
  elif [ -f "$WS_ROOT/.pending_jobs/done/distill-${name}.json" ];         then mark="done   "
  else                                                                         mark="(none) "
  fi

  # Stage row counts (one-line)
  local s1 s2 s3 s4 cur
  s1=$( [ -s "$out_dir/stage1.jsonl"        ] && wc -l < "$out_dir/stage1.jsonl"        || echo 0 )
  s2=$( [ -s "$out_dir/stage2_teacher.jsonl" ] && wc -l < "$out_dir/stage2_teacher.jsonl" || echo 0 )
  s3=$( [ -s "$out_dir/stage3_self.jsonl"    ] && wc -l < "$out_dir/stage3_self.jsonl"    || echo 0 )
  s4=$( [ -s "$out_dir/stage4_audit.jsonl"   ] && wc -l < "$out_dir/stage4_audit.jsonl"   || echo 0 )
  local curated
  curated=$(ls "$out_dir"/curated/*/[a-z]*_distilled.jsonl 2>/dev/null | head -1)
  cur=$( [ -s "$curated" ] && wc -l < "$curated" || echo 0 )

  # Latest tqdm frame (truncated so the line never wraps)
  local frame
  frame=$(latest_tqdm_frame "$log_path")
  # Trim to ~140 cols so wide terminals don't wrap mid-bar
  frame=${frame:0:140}

  printf "● %-28s [%s][%s] s1=%d s2=%d s3=%d s4=%d cur=%d\n" \
    "$name" "$status" "$mark" "$s1" "$s2" "$s3" "$s4" "$cur"
  if [ -n "$frame" ]; then
    printf "    ↳ %s\n" "$frame"
  fi
}

# Renderer selector.
render() {
  if [ "$COMPACT" = 1 ]; then render_one_compact "$1"
  else                        render_one         "$1"
  fi
}

if [ "$COMPACT" = 1 ]; then
  printf "data_gen monitor — workspace=%s — %s\n" "$WS_ROOT" "$(date '+%Y-%m-%d %H:%M:%S')"
  printf "legend: status=[RUN|DONE|FAIL|????]  marker=[pending|done|none]  s1..s4=stage row counts  cur=curated\n"
  echo
fi

# Enumerate names: explicit arg wins; else look at all pending markers.
if [ -n "$PIPELINE" ]; then
  render "$PIPELINE"
  exit 0
fi

shopt -s nullglob
markers=("$WS_ROOT/.pending_jobs"/distill-*.json)
shopt -u nullglob

if [ ${#markers[@]} -eq 0 ]; then
  # Fallback: enumerate any artifacts/generation/<name>/ subdir that
  # has at least a run.log (covers harvested + lingering tmux).
  shopt -s nullglob
  dirs=("$WS_ROOT"/artifacts/generation/*/)
  shopt -u nullglob
  candidates=()
  for d in "${dirs[@]}"; do
    [ -f "$d/run.log" ] && candidates+=("$(basename "$d")")
  done
  if [ ${#candidates[@]} -eq 0 ]; then
    echo "no pipeline runs found under $WS_ROOT/artifacts/generation/"
    exit 0
  fi
  if [ "$COMPACT" != 1 ]; then
    echo "(no pending markers — falling back to artifacts/generation/* with run.log)"
    echo
  fi
  for n in "${candidates[@]}"; do
    render "$n"
    [ "$COMPACT" = 1 ] || echo
  done
else
  if [ "$COMPACT" != 1 ]; then
    echo "found ${#markers[@]} pending pipeline run(s) under $WS_ROOT/.pending_jobs/"
    echo
  fi
  for m in "${markers[@]}"; do
    name=$(basename "$m" .json | sed 's/^distill-//')
    render "$name"
    [ "$COMPACT" = 1 ] || echo
  done
fi
