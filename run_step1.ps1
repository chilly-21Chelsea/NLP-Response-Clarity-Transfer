param(
    [string]$PythonExe = "D:\Important\3自然语言处理\NLP_Assignment\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $Root "src"

& $PythonExe (Join-Path $Root "src\final_pipeline\step1_data_audit.py") `
    --config (Join-Path $Root "configs\step1.json")

if ($LASTEXITCODE -ne 0) {
    throw "Step 1 failed with exit code $LASTEXITCODE"
}
