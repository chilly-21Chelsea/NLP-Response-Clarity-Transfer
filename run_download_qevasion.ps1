param(
    [string]$PythonExe = "D:\Important\3自然语言处理\NLP_Assignment\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $Root "src"

& $PythonExe (Join-Path $Root "src\final_pipeline\download_qevasion.py")

if ($LASTEXITCODE -ne 0) {
    throw "QEvasion download failed with exit code $LASTEXITCODE"
}
