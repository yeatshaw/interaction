$root='D:\LiuY\interaction\LLM4AD'
$env:LLM4AD_CVRP_TEST_DATA='D:\LiuY\dataset\cvrplib\data\cvrp_test_lt200.npz'
& (Join-Path $root 'example\tasks\run_59_common.ps1') `
  -TaskScript (Join-Path $root 'example\tasks\cvrp_construct\run_eoh.py') `
  -LogRoot (Join-Path $root 'logs\cvrp_59')
