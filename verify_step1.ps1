param(
    [string]$PythonExe = "D:\Important\3自然语言处理\NLP_Assignment\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Artifacts = @(
    "data\processed\qevasion_train.parquet",
    "data\processed\qevasion_test.parquet",
    "data\processed\qnli_analysis_train.parquet",
    "data\processed\qnli_analysis_eval.parquet",
    "results\step1\table1_data_statistics.csv",
    "results\step1\label_distribution.csv",
    "results\step1\length_distribution.csv"
)

$Before = @{}
foreach ($RelativePath in $Artifacts) {
    $AbsolutePath = Join-Path $Root $RelativePath
    $Before[$RelativePath] = (Get-FileHash -LiteralPath $AbsolutePath -Algorithm SHA256).Hash
}

& (Join-Path $Root "run_step1.ps1") -PythonExe $PythonExe

$Entries = @()
$MismatchCount = 0
foreach ($RelativePath in $Artifacts) {
    $AbsolutePath = Join-Path $Root $RelativePath
    $After = (Get-FileHash -LiteralPath $AbsolutePath -Algorithm SHA256).Hash
    $Matches = $Before[$RelativePath] -eq $After
    if (-not $Matches) {
        $MismatchCount += 1
    }
    $Entries += [ordered]@{
        path = $RelativePath
        before_sha256 = $Before[$RelativePath]
        after_sha256 = $After
        matches = $Matches
    }
}

$Result = [ordered]@{
    checked_at = (Get-Date).ToString("o")
    status = if ($MismatchCount -eq 0) { "pass" } else { "fail" }
    mismatch_count = $MismatchCount
    files = $Entries
}

$OutputPath = Join-Path $Root "results\step1\reproducibility_check.json"
$Result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding UTF8

if ($MismatchCount -ne 0) {
    throw "Step 1 rerun changed $MismatchCount deterministic artifact(s)"
}

Write-Host "Step 1 reproducibility check: PASS"
Write-Host "Report: $OutputPath"
