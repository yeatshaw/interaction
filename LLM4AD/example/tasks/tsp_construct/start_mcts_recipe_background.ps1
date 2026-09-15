param(
    [string]$PythonExe = "python",
    [string]$TrainData = $env:LLM4AD_TSP_TRAIN_DATA,
    [string]$ExperimentDir = "",
    [string]$LogRoot = ""
)

$ErrorActionPreference = "Stop"

$ScriptPath = Join-Path $PSScriptRoot "run_mcts_recipe.py"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Target script not found: $ScriptPath"
}

Get-Command $PythonExe -ErrorAction Stop | Out-Null

if (-not $LogRoot) {
    $LogRoot = Join-Path $ProjectRoot "logs\background_mcts_recipe_tsp_reflection_recipes"
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $ExperimentDir) {
    $ExperimentDir = Join-Path $ProjectRoot "logs\mcts_recipe_tsp_reflection_recipes_$Timestamp"
}
New-Item -ItemType Directory -Force -Path $ExperimentDir | Out-Null

# Use all Recipe-MCTS recipes and all EoH evolution operators.
$env:LLM4AD_MCTS_RECIPE_MODE = "reflection"
$env:LLM4AD_NO_REFLECTION = "0"
$env:LLM4AD_LOG_DIR = $ExperimentDir
$env:LLM4AD_OPERATORS = "e1,e2,m1,m2"

$StdoutLog = Join-Path $LogRoot "mcts_recipe_tsp_$Timestamp.out.log"
$StderrLog = Join-Path $LogRoot "mcts_recipe_tsp_$Timestamp.err.log"
$PidFile = Join-Path $LogRoot "mcts_recipe_tsp_$Timestamp.pid"

$Arguments = @("-u", "`"$ScriptPath`"")
if ($TrainData) {
    $Arguments += @("--train-data", "`"$TrainData`"")
}

$Process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -LiteralPath $PidFile -Encoding ASCII

[pscustomobject]@{
    Pid = $Process.Id
    Script = $ScriptPath
    WorkingDirectory = $ProjectRoot
    Mode = $env:LLM4AD_MCTS_RECIPE_MODE
    Operators = $env:LLM4AD_OPERATORS
    ExperimentDir = $ExperimentDir
    StdoutLog = $StdoutLog
    StderrLog = $StderrLog
    PidFile = $PidFile
} | Format-List
