#!/usr/bin/env bash
set -u
: "${LLM4AD_TASK_SCRIPT:?Set LLM4AD_TASK_SCRIPT}"
: "${LLM4AD_LOG_ROOT:?Set LLM4AD_LOG_ROOT}"
mkdir -p "$LLM4AD_LOG_ROOT"

launch() {
  local id="$1" comp="$2" attr="$3" summ="$4" atype="$5" stype="$6"
  local identical="$7" shared="$8" population="$9" name
  name=$(printf '%02d' "$id"); mkdir -p "$LLM4AD_LOG_ROOT/$name"
  env LLM4AD_REFLECTION_BITS=11 LLM4AD_RUN_INDEX=0 LLM4AD_REFLECTION_FITNESS=2 \
    LLM4AD_REFLECTION_COMPARISON="$comp" LLM4AD_REFLECTION_ATTRIBUTION="$attr" \
    LLM4AD_REFLECTION_SUMMARY="$summ" LLM4AD_ATTRIBUTION_TYPE="$atype" \
    LLM4AD_SUMMARY_TYPE="$stype" LLM4AD_IDENTICAL_PARENTS="$identical" \
    LLM4AD_SHARED_PARENT="$shared" LLM4AD_POPULATION_COMPARISON="$population" \
    LLM4AD_NUM_SAMPLERS=1 LLM4AD_NUM_EVALUATORS=1 \
    LLM4AD_LOG_DIR="$LLM4AD_LOG_ROOT/$name" \
    nohup python "$LLM4AD_TASK_SCRIPT" >"$LLM4AD_LOG_ROOT/$name.log" \
      2>"$LLM4AD_LOG_ROOT/${name}_error.log" </dev/null &
}

id=0
for parent in 0 1 2 3; do
  for population in elite_worst elite_average worst_average; do
    identical=0; shared=0
    { [ "$parent" -eq 1 ] || [ "$parent" -eq 3 ]; } && identical=1
    { [ "$parent" -eq 2 ] || [ "$parent" -eq 3 ]; } && shared=1
    launch "$id" 1 1 0 both guidance "$identical" "$shared" "$population"
    id=$((id + 1))
  done
done
for comp in 0 1; do for attr in 0 1; do for summ in 0 1; do
  [ "$comp$attr$summ" = 000 ] && continue
  [ "$attr" -eq 1 ] && attrs='good bad both difference same' || attrs='good'
  [ "$summ" -eq 1 ] && sums='guidance experience conditions' || sums='guidance'
  for atype in $attrs; do for stype in $sums; do
    launch "$id" "$comp" "$attr" "$summ" "$atype" "$stype" 1 1 elite_worst
    id=$((id + 1))
  done; done
done; done; done
echo "Started $id experiments."
