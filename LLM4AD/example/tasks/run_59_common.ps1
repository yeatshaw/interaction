param([Parameter(Mandatory=$true)][string]$TaskScript,
      [Parameter(Mandatory=$true)][string]$LogRoot)
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
function Start-One($id,$comp,$attr,$summ,$atype,$stype,$identical,$shared,$population) {
  $name='{0:D2}' -f $id; $dir=Join-Path $LogRoot $name
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $env:LLM4AD_REFLECTION_BITS='11'; $env:LLM4AD_RUN_INDEX='0'; $env:LLM4AD_REFLECTION_FITNESS='2'
  $env:LLM4AD_REFLECTION_COMPARISON="$comp"; $env:LLM4AD_REFLECTION_ATTRIBUTION="$attr"
  $env:LLM4AD_REFLECTION_SUMMARY="$summ"; $env:LLM4AD_ATTRIBUTION_TYPE=$atype
  $env:LLM4AD_SUMMARY_TYPE=$stype; $env:LLM4AD_IDENTICAL_PARENTS="$identical"
  $env:LLM4AD_SHARED_PARENT="$shared"; $env:LLM4AD_POPULATION_COMPARISON=$population
  $env:LLM4AD_NUM_SAMPLERS='1'; $env:LLM4AD_NUM_EVALUATORS='1'; $env:LLM4AD_LOG_DIR=$dir
  Start-Process python -ArgumentList "`"$TaskScript`"" -WorkingDirectory (Split-Path (Split-Path (Split-Path $TaskScript))) `
    -RedirectStandardOutput (Join-Path $LogRoot "$name.log") `
    -RedirectStandardError (Join-Path $LogRoot "${name}_error.log") -WindowStyle Hidden
}
$id=0
foreach($parent in 0..3){ foreach($population in @('elite_worst','elite_average','worst_average')){
  $identical=[int]($parent -eq 1 -or $parent -eq 3); $shared=[int]($parent -eq 2 -or $parent -eq 3)
  Start-One $id 1 1 0 'both' 'guidance' $identical $shared $population; $id++
}}
$atypes=@('good','bad','both','difference','same'); $stypes=@('guidance','experience','conditions')
foreach($comp in 0..1){foreach($attr in 0..1){foreach($summ in 0..1){
  if(-not($comp -or $attr -or $summ)){continue}
  $aa=if($attr){$atypes}else{@('good')}; $ss=if($summ){$stypes}else{@('guidance')}
  foreach($a in $aa){foreach($s in $ss){Start-One $id $comp $attr $summ $a $s 1 1 'elite_worst'; $id++}}
}}}
Write-Host "Started $id experiments."
