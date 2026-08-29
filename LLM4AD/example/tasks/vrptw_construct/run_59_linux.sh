#!/usr/bin/env bash
set -u

ROOT="/public/home/liuyang/interaction/LLM4AD"
SCRIPT="$ROOT/example/tasks/vrptw_construct/run_eoh.py"
LOGROOT="$ROOT/logs/vrptw_59"
mkdir -p "$LOGROOT"

run_one() {
  local id="$1" config="$2" run="$3"
  local comp="$4" attr="$5" summ="$6" attr_type="$7" summ_type="$8"
  local identical="$9" shared="${10}" pop="${11}"
  local name
  name=$(printf '%02d' "$id")
  mkdir -p "$LOGROOT/$name"
  (
    export LLM4AD_REFLECTION_BITS=11
    export LLM4AD_RUN_INDEX="$run"
    export LLM4AD_REFLECTION_FITNESS=0
    export LLM4AD_REFLECTION_COMPARISON="$comp"
    export LLM4AD_REFLECTION_ATTRIBUTION="$attr"
    export LLM4AD_REFLECTION_SUMMARY="$summ"
    export LLM4AD_ATTRIBUTION_TYPE="$attr_type"
    export LLM4AD_SUMMARY_TYPE="$summ_type"
    export LLM4AD_IDENTICAL_PARENTS="$identical"
    export LLM4AD_SHARED_PARENT="$shared"
    export LLM4AD_POPULATION_COMPARISON="$pop"
    export LLM4AD_LOG_DIR="$LOGROOT/$name"
    python "$SCRIPT" >"$LOGROOT/${name}.log" 2>&1
  ) &
}

id=0
# Experiment 1: 4 parent configurations x 3 population comparisons.
for parent in 0 1 2 3; do
  for pop in elite_worst elite_average worst_average; do
    identical=0; shared=0
    [ "$parent" -eq 1 ] || [ "$parent" -eq 3 ] && identical=1
    [ "$parent" -eq 2 ] || [ "$parent" -eq 3 ] && shared=1
    run_one "$id" "$((id / 1))" 0 1 1 0 both guidance "$identical" "$shared" "$pop"
    id=$((id + 1))
  done
done

# Experiment 2: fixed fourth parent configuration; 47 method configurations.
for comp in 0 1; do
  for attr in 0 1; do
    for summ in 0 1; do
      [ "$comp$attr$summ" = 000 ] && continue
      if [ "$attr" -eq 1 ]; then attrs="good bad both difference same"; else attrs="good"; fi
      if [ "$summ" -eq 1 ]; then sums="guidance experience conditions"; else sums="guidance"; fi
      for attr_type in $attrs; do
        for summ_type in $sums; do
          run_one "$id" "$id" 0 "$comp" "$attr" "$summ" "$attr_type" "$summ_type" 1 1 elite_worst
          id=$((id + 1))
        done
      done
    done
  done
done
wait
