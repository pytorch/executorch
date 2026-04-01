<#
.SYNOPSIS
Benchmark the ExecuTorch Parakeet C++ runner on Windows.

.DESCRIPTION
Runs parakeet_runner.exe multiple times, extracts the machine-readable
"PyTorchObserver { ... }" metrics emitted by the runner, and prints per-run
and aggregate latency/throughput summaries.

.EXAMPLE
.\examples\models\parakeet\benchmark_parakeet_runner.ps1 `
  -ModelPath C:\models\parakeet\model.pte `
  -AudioPath C:\models\parakeet\sample.wav `
  -TokenizerPath C:\models\parakeet\tokenizer.model `
  -DataPath C:\models\parakeet\aoti_cuda_blob.ptd `
  -Runs 20 -Warmup 3

.EXAMPLE
.\examples\models\parakeet\benchmark_parakeet_runner.ps1 `
  -ModelPath C:\models\parakeet\model.pte `
  -AudioPath C:\models\parakeet\sample.wav `
  -TokenizerPath C:\models\parakeet\tokenizer.model `
  -ExePath C:\executorch\cmake-out\examples\models\parakeet\Release\parakeet_runner.exe `
  -CsvPath C:\tmp\parakeet_bench.csv
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$AudioPath,

    [Parameter(Mandatory = $true)]
    [string]$TokenizerPath,

    [string]$DataPath = "",

    [string]$ExePath = "",

    [ValidateRange(0, 1000)]
    [int]$Warmup = 1,

    [ValidateRange(1, 10000)]
    [int]$Runs = 10,

    [ValidateSet("none", "token", "word", "segment", "all")]
    [string]$Timestamps = "none",

    [string[]]$ExtraArgs = @(),

    [string]$CsvPath = "",

    [switch]$PrintRunnerOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-DefaultExePath {
    param([string]$ScriptRoot)

    $repoRoot = (Resolve-Path (Join-Path $ScriptRoot "..\..\..")).Path
    $candidates = @(
        (Join-Path $repoRoot "cmake-out\examples\models\parakeet\Release\parakeet_runner.exe"),
        (Join-Path $repoRoot "cmake-out\examples\models\parakeet\parakeet_runner.exe"),
        (Join-Path $repoRoot "cmake-out-release\examples\models\parakeet\Release\parakeet_runner.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return ""
}

function Get-Stats {
    param([double[]]$Values)

    if (-not $Values -or $Values.Count -eq 0) {
        return $null
    }

    $count = [double]$Values.Count
    $sum = 0.0
    $min = [double]::PositiveInfinity
    $max = [double]::NegativeInfinity
    foreach ($v in $Values) {
        $sum += $v
        if ($v -lt $min) { $min = $v }
        if ($v -gt $max) { $max = $v }
    }

    $mean = $sum / $count
    $variance = 0.0
    foreach ($v in $Values) {
        $d = $v - $mean
        $variance += ($d * $d)
    }
    $stddev = [math]::Sqrt($variance / $count)

    return [pscustomobject]@{
        Count  = [int]$count
        Mean   = $mean
        StdDev = $stddev
        Min    = $min
        Max    = $max
    }
}

function Get-ObserverJson {
    param([string[]]$Lines)

    $jsonLine = $null
    foreach ($line in $Lines) {
        if ($line -match "^PyTorchObserver\s+(\{.*\})\s*$") {
            $jsonLine = $Matches[1]
        }
    }
    if (-not $jsonLine) {
        return $null
    }
    return ($jsonLine | ConvertFrom-Json)
}

function Invoke-ParakeetOnce {
    param(
        [string]$RunnerPath,
        [string[]]$RunnerArgs,
        [switch]$PrintOutput
    )

    $start = Get-Date
    $lines = & $RunnerPath @RunnerArgs 2>&1 | ForEach-Object { $_.ToString() }
    $exitCode = $LASTEXITCODE
    $wallMs = ((Get-Date) - $start).TotalMilliseconds

    if ($PrintOutput) {
        foreach ($line in $lines) {
            Write-Host $line
        }
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Lines    = $lines
        WallMs   = [double]$wallMs
    }
}

function New-SummaryRow {
    param(
        [string]$Metric,
        [double[]]$Values
    )

    $s = Get-Stats -Values $Values
    if ($null -eq $s) {
        return [pscustomobject]@{
            Metric = $Metric
            Mean   = $null
            StdDev = $null
            Min    = $null
            Max    = $null
        }
    }

    return [pscustomobject]@{
        Metric = $Metric
        Mean   = [math]::Round($s.Mean, 3)
        StdDev = [math]::Round($s.StdDev, 3)
        Min    = [math]::Round($s.Min, 3)
        Max    = [math]::Round($s.Max, 3)
    }
}

if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $ExePath = Get-DefaultExePath -ScriptRoot $PSScriptRoot
}
if ([string]::IsNullOrWhiteSpace($ExePath)) {
    throw "Unable to auto-locate parakeet_runner.exe. Pass -ExePath explicitly."
}

$requiredPaths = @(
    @{ Name = "ExePath"; Value = $ExePath },
    @{ Name = "ModelPath"; Value = $ModelPath },
    @{ Name = "AudioPath"; Value = $AudioPath },
    @{ Name = "TokenizerPath"; Value = $TokenizerPath }
)
if (-not [string]::IsNullOrWhiteSpace($DataPath)) {
    $requiredPaths += @{ Name = "DataPath"; Value = $DataPath }
}
foreach ($p in $requiredPaths) {
    if (-not (Test-Path $p.Value)) {
        throw "$($p.Name) does not exist: $($p.Value)"
    }
}

$baseArgs = @(
    "--model_path", $ModelPath,
    "--audio_path", $AudioPath,
    "--tokenizer_path", $TokenizerPath,
    "--timestamps", $Timestamps
)
if (-not [string]::IsNullOrWhiteSpace($DataPath)) {
    $baseArgs += @("--data_path", $DataPath)
}
if ($ExtraArgs.Count -gt 0) {
    $baseArgs += $ExtraArgs
}

Write-Host "Benchmark config:"
Write-Host "  ExePath   : $ExePath"
Write-Host "  Runs      : $Runs"
Write-Host "  Warmup    : $Warmup"
Write-Host "  Timestamps: $Timestamps"
Write-Host "  Args      : $($baseArgs -join ' ')"
Write-Host ""

for ($i = 1; $i -le $Warmup; $i++) {
    Write-Host ("Warmup {0}/{1}..." -f $i, $Warmup)
    $run = Invoke-ParakeetOnce -RunnerPath $ExePath -RunnerArgs $baseArgs -PrintOutput:$PrintRunnerOutput
    if ($run.ExitCode -ne 0) {
        throw "Warmup run failed with exit code $($run.ExitCode).`n$($run.Lines -join [Environment]::NewLine)"
    }
}

$results = @()
for ($i = 1; $i -le $Runs; $i++) {
    Write-Host ("Run {0}/{1}..." -f $i, $Runs)
    $run = Invoke-ParakeetOnce -RunnerPath $ExePath -RunnerArgs $baseArgs -PrintOutput:$PrintRunnerOutput
    if ($run.ExitCode -ne 0) {
        throw "Benchmark run failed with exit code $($run.ExitCode).`n$($run.Lines -join [Environment]::NewLine)"
    }

    $obs = Get-ObserverJson -Lines $run.Lines
    if ($null -eq $obs) {
        throw "Could not find 'PyTorchObserver { ... }' line in runner output for run $i.`n$($run.Lines -join [Environment]::NewLine)"
    }

    $modelLoadMs = [double]$obs.model_load_end_ms - [double]$obs.model_load_start_ms
    $inferenceMs = [double]$obs.inference_end_ms - [double]$obs.inference_start_ms
    $promptEvalMs = [double]$obs.prompt_eval_end_ms - [double]$obs.inference_start_ms
    $decodeMs = [double]$obs.inference_end_ms - [double]$obs.prompt_eval_end_ms
    $ttftMs = [double]$obs.first_token_ms - [double]$obs.inference_start_ms
    $promptTokens = [double]$obs.prompt_tokens
    $generatedTokens = [double]$obs.generated_tokens

    $totalTps = if ($inferenceMs -gt 0) { $generatedTokens * 1000.0 / $inferenceMs } else { 0.0 }
    $decodeTps = if ($decodeMs -gt 0) { $generatedTokens * 1000.0 / $decodeMs } else { 0.0 }
    $promptTps = if ($promptEvalMs -gt 0) { $promptTokens * 1000.0 / $promptEvalMs } else { 0.0 }

    $results += [pscustomobject]@{
        Iteration       = $i
        ModelLoadMs     = [math]::Round($modelLoadMs, 3)
        InferenceMs     = [math]::Round($inferenceMs, 3)
        PromptEvalMs    = [math]::Round($promptEvalMs, 3)
        DecodeMs        = [math]::Round($decodeMs, 3)
        TTFTMs          = [math]::Round($ttftMs, 3)
        PromptTokens    = [math]::Round($promptTokens, 3)
        GeneratedTokens = [math]::Round($generatedTokens, 3)
        TotalTPS        = [math]::Round($totalTps, 3)
        DecodeTPS       = [math]::Round($decodeTps, 3)
        PromptTPS       = [math]::Round($promptTps, 3)
        WallClockMs     = [math]::Round($run.WallMs, 3)
    }
}

Write-Host ""
Write-Host "Per-run metrics:"
$results | Format-Table -AutoSize

$summary = @(
    (New-SummaryRow -Metric "ModelLoadMs" -Values ($results | ForEach-Object { [double]$_.ModelLoadMs })),
    (New-SummaryRow -Metric "InferenceMs" -Values ($results | ForEach-Object { [double]$_.InferenceMs })),
    (New-SummaryRow -Metric "PromptEvalMs" -Values ($results | ForEach-Object { [double]$_.PromptEvalMs })),
    (New-SummaryRow -Metric "DecodeMs" -Values ($results | ForEach-Object { [double]$_.DecodeMs })),
    (New-SummaryRow -Metric "TTFTMs" -Values ($results | ForEach-Object { [double]$_.TTFTMs })),
    (New-SummaryRow -Metric "TotalTPS" -Values ($results | ForEach-Object { [double]$_.TotalTPS })),
    (New-SummaryRow -Metric "DecodeTPS" -Values ($results | ForEach-Object { [double]$_.DecodeTPS })),
    (New-SummaryRow -Metric "WallClockMs" -Values ($results | ForEach-Object { [double]$_.WallClockMs }))
)

Write-Host ""
Write-Host "Summary (mean/std/min/max):"
$summary | Format-Table -AutoSize

if (-not [string]::IsNullOrWhiteSpace($CsvPath)) {
    $results | Export-Csv -Path $CsvPath -NoTypeInformation
    Write-Host ""
    Write-Host "Saved per-run metrics to: $CsvPath"
}
