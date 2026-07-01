param(
  [string]$ModelDir = "models/VoxCPM2",
  [string]$Device = "cuda",
  [string]$OutDir = "output/accepted-baseline",
  [string]$PromptSet = "tests/golden/mandarin_regression_prompts.json",
  [string]$RefAudio = "",
  [int]$Steps = 30,
  [int]$Seed = 99,
  [switch]$DryRun,
  [switch]$SkipClone,
  [switch]$RunAsr,
  [string]$AsrRunner = "$env:USERPROFILE\.codex\skills\asr-transcribe\scripts\run-asr.ps1",
  [string]$AsrEngine = "faster-whisper",
  [string]$AsrModel = "large-v3-turbo",
  [switch]$UpdateAcceptanceReport
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($DryRun -and $RunAsr) {
  throw "ASR baseline is only meaningful for real model outputs; remove -DryRun or omit -RunAsr."
}

function ConvertTo-RelativePath([string]$Path) {
  $resolved = Resolve-Path -LiteralPath $Path
  $rootPath = (Resolve-Path -LiteralPath $root).Path
  return [IO.Path]::GetRelativePath($rootPath, $resolved.Path)
}

function Get-GitValue([string[]]$GitArgs) {
  $value = & git @GitArgs
  if ($LASTEXITCODE -ne 0) {
    throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE"
  }
  return ($value -join "`n").Trim()
}

function Read-JsonFile([string]$Path) {
  return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-PromptCaseMap([string]$Path) {
  $spec = Read-JsonFile -Path $Path
  $map = @{}
  foreach ($case in $spec.cases) {
    $map[$case.id] = $case
  }
  return $map
}

function Invoke-Asr([string]$WavPath, [string]$CaseId, [string[]]$RequiredTerms) {
  if (-not (Test-Path -LiteralPath $AsrRunner -PathType Leaf)) {
    throw "ASR runner not found: $AsrRunner"
  }

  $asrRoot = Join-Path $outRoot.FullName "asr"
  $caseOut = New-Item -ItemType Directory -Force -Path (Join-Path $asrRoot $CaseId)
  Write-Host ""
  Write-Host ">>> ASR $CaseId" -ForegroundColor Cyan
  powershell -ExecutionPolicy Bypass -File $AsrRunner `
    -InputMedia $WavPath `
    -OutputDir $caseOut.FullName `
    -Engine $AsrEngine `
    -Model $AsrModel
  if ($LASTEXITCODE -ne 0) {
    throw "ASR failed for $CaseId with exit code $LASTEXITCODE"
  }

  $transcriptPath = Join-Path $caseOut.FullName "transcript.txt"
  if (-not (Test-Path -LiteralPath $transcriptPath -PathType Leaf)) {
    throw "missing ASR transcript for ${CaseId}: $transcriptPath"
  }

  $transcript = (Get-Content -LiteralPath $transcriptPath -Raw).Trim()
  $missing = @()
  foreach ($term in $RequiredTerms) {
    if (-not $transcript.Contains($term)) {
      $missing += $term
    }
  }

  return [ordered]@{
    transcript_path = ConvertTo-RelativePath -Path $transcriptPath
    manifest_path = ConvertTo-RelativePath -Path (Join-Path $caseOut.FullName "asr_manifest.json")
    transcript = $transcript
    required_terms = $RequiredTerms
    missing_terms = $missing
    passed = ($missing.Count -eq 0)
  }
}

function New-ReportMarkdown($Manifest) {
  $lines = [Collections.Generic.List[string]]::new()
  $lines.Add("# VoxCPM2 Accepted Baseline")
  $lines.Add("")
  $lines.Add("- generated_at: $($Manifest.generated_at)")
  $lines.Add("- commit: $($Manifest.commit)")
  $lines.Add("- device: $($Manifest.device)")
  $lines.Add("- dry_run: $($Manifest.dry_run)")
  $lines.Add("- status: $($Manifest.acceptance_status)")
  $lines.Add("")
  $lines.Add("## Command")
  $lines.Add("")
  $lines.Add('```powershell')
  $lines.Add($Manifest.command)
  $lines.Add('```')
  $lines.Add("")
  $lines.Add("## Cases")
  $lines.Add("")
  $lines.Add("| case | clone | wav | peak | high_zcr_frames | ASR |")
  $lines.Add("| --- | --- | --- | --- | --- | --- |")
  foreach ($case in $Manifest.cases) {
    $asrText = "not-run"
    if ($null -ne $case.asr) {
      $asrText = if ($case.asr.passed) { "pass" } else { "fail" }
    }
    $lines.Add((
      "| {0} | {1} | {2} | {3:n4} | {4} | {5} |" -f
      $case.id,
      $case.is_clone,
      $case.wav_path,
      [double]$case.quality.peak,
      [int]$case.quality.high_zcr_frame_count,
      $asrText
    ))
  }
  $lines.Add("")
  $lines.Add("## Notes")
  $lines.Add("")
  if ($Manifest.dry_run) {
    $lines.Add("- This is a dry-run harness validation, not an accepted real-model speech baseline.")
  } elseif (-not $Manifest.asr_enabled) {
    $lines.Add("- Real-model audio metrics were collected, but ASR was not run. Run again with -RunAsr before marking this baseline accepted.")
  } else {
    $lines.Add("- ASR required-term checks were run for every generated case.")
  }
  $lines.Add("- `baseline_manifest.json` is the machine-readable source of truth for this report.")
  return ($lines -join "`r`n") + "`r`n"
}

function Update-AcceptanceReportFile([string]$BaselineReportPath, [string]$ManifestPath, [string]$Status) {
  $reportPath = Join-Path $root "acceptance_report.md"
  $start = "<!-- ACCEPTANCE_BASELINE_START -->"
  $end = "<!-- ACCEPTANCE_BASELINE_END -->"
  $block = @"
$start
### Latest Accepted Baseline Artifact

**Status:** $Status

- Baseline report: `$(ConvertTo-RelativePath -Path $BaselineReportPath)`
- Baseline manifest: `$(ConvertTo-RelativePath -Path $ManifestPath)`
- Rebuild command: `harness\write_acceptance_baseline.ps1`

$end
"@

  $existing = ""
  if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
    $existing = Get-Content -LiteralPath $reportPath -Raw
  }

  $pattern = [regex]::Escape($start) + ".*?" + [regex]::Escape($end)
  if ($existing -match $pattern) {
    $updated = [regex]::Replace($existing, $pattern, $block, [Text.RegularExpressions.RegexOptions]::Singleline)
  } else {
    $updated = $existing.TrimEnd() + "`r`n`r`n" + $block + "`r`n"
  }
  Set-Content -LiteralPath $reportPath -Value $updated -Encoding UTF8
}

$outRoot = New-Item -ItemType Directory -Force -Path $OutDir
$promptPath = Resolve-Path -LiteralPath $PromptSet
$caseMap = Get-PromptCaseMap -Path $promptPath.Path
$commit = Get-GitValue -GitArgs @("rev-parse", "HEAD")
$generatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

$gateParams = @{
  ModelDir = $ModelDir
  Device = $Device
  OutDir = $outRoot.FullName
  PromptSet = $promptPath.Path
  Steps = $Steps
  Seed = $Seed
}
if ($DryRun) { $gateParams["DryRun"] = $true }
if ($SkipClone) { $gateParams["SkipClone"] = $true }
if (-not [string]::IsNullOrWhiteSpace($RefAudio)) {
  $gateParams["RefAudio"] = $RefAudio
}

$gateArgsForDisplay = [Collections.Generic.List[string]]::new()
foreach ($key in @("ModelDir", "Device", "OutDir", "PromptSet", "RefAudio", "Steps", "Seed")) {
  if ($gateParams.ContainsKey($key)) {
    $value = [string]$gateParams[$key]
    if ($value -match "\s") { $value = "`"$value`"" }
    $gateArgsForDisplay.Add("-$key")
    $gateArgsForDisplay.Add($value)
  }
}
foreach ($switchName in @("DryRun", "SkipClone")) {
  if ($gateParams.ContainsKey($switchName)) {
    $gateArgsForDisplay.Add("-$switchName")
  }
}
$gateCommand = ".\harness\audio_quality_gate.ps1 " + ($gateArgsForDisplay -join " ")

Write-Host ""
Write-Host ">>> $gateCommand" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "audio_quality_gate.ps1") @gateParams
if ($LASTEXITCODE -ne 0) {
  throw "audio quality gate failed with exit code $LASTEXITCODE"
}

$cases = [Collections.Generic.List[object]]::new()
$allAsrPassed = $true
foreach ($metricsFile in Get-ChildItem -LiteralPath $outRoot.FullName -Filter "*.metrics.json" | Sort-Object Name) {
  $id = $metricsFile.BaseName -replace "\.metrics$", ""
  $metrics = Read-JsonFile -Path $metricsFile.FullName
  $qualityPath = [IO.Path]::ChangeExtension($metricsFile.FullName, ".quality.json")
  if (-not (Test-Path -LiteralPath $qualityPath -PathType Leaf)) {
    throw "missing quality sidecar: $qualityPath"
  }
  $quality = Read-JsonFile -Path $qualityPath

  $requiredTerms = @()
  if ($caseMap.ContainsKey($id)) {
    $requiredTerms = @($caseMap[$id].required_terms)
  } elseif ([bool]$metrics.is_clone) {
    $requiredTerms = @("声音复制", "噪声回归")
  }

  $asr = $null
  if ($RunAsr) {
    $asr = Invoke-Asr -WavPath $quality.wav_path -CaseId $id -RequiredTerms $requiredTerms
    if (-not $asr.passed) {
      $allAsrPassed = $false
    }
  }

  $cases.Add([ordered]@{
    id = $id
    text = if ($caseMap.ContainsKey($id)) { $caseMap[$id].text } elseif ([bool]$metrics.is_clone) { "这是声音复制的噪声回归测试。" } else { $null }
    is_clone = [bool]$metrics.is_clone
    wav_path = ConvertTo-RelativePath -Path $quality.wav_path
    metrics_path = ConvertTo-RelativePath -Path $metricsFile.FullName
    quality_path = ConvertTo-RelativePath -Path $qualityPath
    metrics = $metrics
    quality = $quality
    asr = $asr
  })
}

if ($cases.Count -eq 0) {
  throw "no metrics files found under $($outRoot.FullName)"
}

$status = if ($DryRun) {
  "dry_run_only"
} elseif (-not $RunAsr) {
  "needs_asr"
} elseif ($allAsrPassed) {
  "accepted"
} else {
  "asr_failed"
}

$manifest = [ordered]@{
  schema_version = 1
  generated_at = $generatedAt
  commit = $commit
  model_dir = $ModelDir
  device = $Device
  steps = $Steps
  seed = $Seed
  dry_run = [bool]$DryRun
  asr_enabled = [bool]$RunAsr
  asr_engine = if ($RunAsr) { $AsrEngine } else { $null }
  asr_model = if ($RunAsr) { $AsrModel } else { $null }
  acceptance_status = $status
  command = $gateCommand
  cases = $cases
}

$manifestPath = Join-Path $outRoot.FullName "baseline_manifest.json"
$reportPath = Join-Path $outRoot.FullName "baseline_report.md"
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
New-ReportMarkdown -Manifest $manifest | Set-Content -LiteralPath $reportPath -Encoding UTF8

if ($UpdateAcceptanceReport) {
  Update-AcceptanceReportFile -BaselineReportPath $reportPath -ManifestPath $manifestPath -Status $status
}

Write-Host ""
Write-Host "Baseline manifest: $manifestPath" -ForegroundColor Green
Write-Host "Baseline report:   $reportPath" -ForegroundColor Green
Write-Host "Status:            $status" -ForegroundColor Green
