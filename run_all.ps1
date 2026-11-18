param(
    [string]$PythonExe = "D:\Important\3自然语言处理\NLP_Assignment\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath (Join-Path $Root "data\qevasion\train_clean.csv"))) {
    & (Join-Path $Root "run_download_qevasion.ps1") -PythonExe $PythonExe
}
& (Join-Path $Root "run_step1.ps1") -PythonExe $PythonExe
& (Join-Path $Root "run_step2.ps1") -PythonExe $PythonExe
& (Join-Path $Root "run_step3.ps1") -PythonExe $PythonExe
& (Join-Path $Root "run_step4.ps1") -PythonExe $PythonExe

$ReportPython = $PythonExe
$env:NLP_OUTPUTS = Join-Path $Root "outputs"
& $ReportPython (Join-Path $Root "src\final_pipeline\build_reports.py")

if ($LASTEXITCODE -ne 0) {
    throw "Report build failed with exit code $LASTEXITCODE"
}

Write-Host "All experiments and reports completed."
