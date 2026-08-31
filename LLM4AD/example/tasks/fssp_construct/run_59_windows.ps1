$root='D:\LiuY\interaction\LLM4AD'
$env:LLM4AD_FSSP_TEST_DATA='D:\LiuY\dataset\Taillard_scheduling\fsp\fsp_instances.npz'
& (Join-Path $root 'example\tasks\run_59_common.ps1') `
  -TaskScript (Join-Path $root 'example\tasks\fssp_construct\run_eoh.py') `
  -LogRoot (Join-Path $root 'logs\fssp_59')
