param(
    [string]$PythonExe = "python",
    [string]$TaskScript = (Join-Path $PSScriptRoot "run_mcts_recipe.py"),
    [string]$LogDir = ""
)

$ErrorActionPreference = "Stop"

$TaskScript = (Resolve-Path -LiteralPath $TaskScript).Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path

if (-not $LogDir) {
    $LogDir = Join-Path $RepoRoot "logs\background"
}
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdOut = Join-Path $LogDir "cvrp_mcts_recipe_$Timestamp.log"
$StdErr = Join-Path $LogDir "cvrp_mcts_recipe_${Timestamp}_error.log"

$Process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("`"$TaskScript`"") `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdOut `
    -RedirectStandardError $StdErr `
    -PassThru

$LaunchInfo = [ordered]@{
    pid = $Process.Id
    python = $PythonExe
    task_script = $TaskScript
    working_directory = $RepoRoot
    stdout_log = $StdOut
    stderr_log = $StdErr
    config_file = (Join-Path $PSScriptRoot "mcts_recipe_config.py")
    started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

$LaunchJson = Join-Path $LogDir "cvrp_mcts_recipe_${Timestamp}_launch.json"
$LaunchInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $LaunchJson -Encoding UTF8

Write-Host "Started CVRP Recipe-MCTS background run."
Write-Host "PID: $($Process.Id)"
Write-Host "Config: $($LaunchInfo.config_file)"
Write-Host "Stdout: $StdOut"
Write-Host "Stderr: $StdErr"
Write-Host "Launch JSON: $LaunchJson"
