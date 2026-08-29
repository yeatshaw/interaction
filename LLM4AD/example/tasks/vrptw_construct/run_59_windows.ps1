$root = 'D:\LiuY\interaction\LLM4AD'
$script = Join-Path $root 'example\tasks\vrptw_construct\run_eoh.py'
$logRoot = Join-Path $root 'logs\vrptw_59'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Start-Experiment($id, $comp, $attr, $summ, $attrType, $summType, $identical, $shared, $pop) {
    $name = '{0:D2}' -f $id
    $dir = Join-Path $logRoot $name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:LLM4AD_REFLECTION_BITS = '11'
    $env:LLM4AD_RUN_INDEX = '0'
    $env:LLM4AD_REFLECTION_FITNESS = '0'
    $env:LLM4AD_REFLECTION_COMPARISON = "$comp"
    $env:LLM4AD_REFLECTION_ATTRIBUTION = "$attr"
    $env:LLM4AD_REFLECTION_SUMMARY = "$summ"
    $env:LLM4AD_ATTRIBUTION_TYPE = $attrType
    $env:LLM4AD_SUMMARY_TYPE = $summType
    $env:LLM4AD_IDENTICAL_PARENTS = "$identical"
    $env:LLM4AD_SHARED_PARENT = "$shared"
    $env:LLM4AD_POPULATION_COMPARISON = $pop
    $env:LLM4AD_LOG_DIR = $dir
    Start-Process python -ArgumentList "`"$script`"" -WorkingDirectory $root `
      -RedirectStandardOutput (Join-Path $logRoot "$name.log") `
      -RedirectStandardError (Join-Path $logRoot "${name}_error.log") -WindowStyle Hidden
}

$id = 0
foreach ($parent in 0..3) {
  foreach ($pop in @('elite_worst','elite_average','worst_average')) {
    $identical = [int]($parent -eq 1 -or $parent -eq 3)
    $shared = [int]($parent -eq 2 -or $parent -eq 3)
    Start-Experiment $id 1 1 0 'both' 'guidance' $identical $shared $pop
    $id++
  }
}

$attrTypes = @('good','bad','both','difference','same')
$sumTypes = @('guidance','experience','conditions')
foreach ($comp in 0..1) {
  foreach ($attr in 0..1) {
    foreach ($summ in 0..1) {
      if ($comp -eq 0 -and $attr -eq 0 -and $summ -eq 0) { continue }
      $as = if ($attr) { $attrTypes } else { @('good') }
      $ss = if ($summ) { $sumTypes } else { @('guidance') }
      foreach ($a in $as) { foreach ($s in $ss) {
        Start-Experiment $id $comp $attr $summ $a $s 1 1 'elite_worst'
        $id++
      }}
    }
  }
}
Write-Host "Started $id experiments."
