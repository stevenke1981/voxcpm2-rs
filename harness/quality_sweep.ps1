param(
  [string]$ModelDir = "models/VoxCPM2",
  [string]$Device = "cuda",
  [string]$OutDir = "output/quality-sweep",
  [string]$PromptSet = "tests/golden/mandarin_regression_prompts.json",
  [string]$RefAudio = "",
  [int[]]$Seeds = @(99, 100, 101, 102),
  [double[]]$CfgValues = @(2.0, 2.3, 2.5, 2.7),
  [string[]]$LatentNormValues = @("", "0.72", "0.7875", "0.85"),
  [string[]]$Schedulers = @("uniform", "log-norm"),
  [int]$Steps = 30,
  [string[]]$CaseIds = @(),
  [int]$MaxCombos = 0,
  [switch]$DryRun,
  [switch]$SkipClone,
  [switch]$RunAsr,
  [string]$AsrRunner = "$env:USERPROFILE\.codex\skills\asr-transcribe\scripts\run-asr.ps1",
  [string]$AsrEngine = "faster-whisper",
  [string]$AsrModel = "large-v3-turbo"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($DryRun -and $RunAsr) {
  throw "ASR scoring is only meaningful for real model outputs; remove -DryRun or omit -RunAsr."
}

function Get-CargoFeatureArgs([string]$DeviceName) {
  switch ($DeviceName.ToLowerInvariant()) {
    "cuda" { return @("--no-default-features", "--features", "cuda") }
    "metal" { return @("--no-default-features", "--features", "metal") }
    "mkl" { return @("--no-default-features", "--features", "mkl") }
    default { return @("--features", "cpu") }
  }
}

function ConvertTo-RelativePath([string]$Path) {
  $resolved = Resolve-Path -LiteralPath $Path
  $rootPath = (Resolve-Path -LiteralPath $root).Path
  return [IO.Path]::GetRelativePath($rootPath, $resolved.Path)
}

function ConvertTo-SafeId([string]$Value) {
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return "default"
  }
  return ($Value -replace "[^A-Za-z0-9]+", "p").Trim("p")
}

function Read-JsonFile([string]$Path) {
  return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-NormalizedText([string]$Text) {
  if ($null -eq $Text) { return "" }
  $withoutTimestamps = [regex]::Replace($Text, "\[[0-9:.]+\]", "")
  $normalized = [regex]::Replace($withoutTimestamps, "[\s\p{P}\p{S}]+", "")
  return $normalized.ToLowerInvariant()
}

function Get-EditDistance([string]$A, [string]$B) {
  $n = $A.Length
  $m = $B.Length
  $prev = New-Object int[] ($m + 1)
  $curr = New-Object int[] ($m + 1)
  for ($j = 0; $j -le $m; $j++) { $prev[$j] = $j }
  for ($i = 1; $i -le $n; $i++) {
    $curr[0] = $i
    for ($j = 1; $j -le $m; $j++) {
      $cost = if ($A[$i - 1] -eq $B[$j - 1]) { 0 } else { 1 }
      $delete = $prev[$j] + 1
      $insert = $curr[$j - 1] + 1
      $replace = $prev[$j - 1] + $cost
      $curr[$j] = [Math]::Min([Math]::Min($delete, $insert), $replace)
    }
    $tmp = $prev
    $prev = $curr
    $curr = $tmp
  }
  return $prev[$m]
}

function Get-TextSimilarity([string]$Expected, [string]$Actual) {
  $expectedNorm = Get-NormalizedText -Text $Expected
  $actualNorm = Get-NormalizedText -Text $Actual
  $maxLen = [Math]::Max($expectedNorm.Length, $actualNorm.Length)
  if ($maxLen -eq 0) { return 1.0 }
  $distance = Get-EditDistance -A $expectedNorm -B $actualNorm
  return [Math]::Max(0.0, 1.0 - ([double]$distance / [double]$maxLen))
}

function Read-Pcm16WavMetrics([string]$Path, [double]$FrameMs, [double]$HighZcrThreshold, [double]$HarshRmsThreshold) {
  $stream = [IO.File]::OpenRead((Resolve-Path -LiteralPath $Path))
  $reader = [IO.BinaryReader]::new($stream)
  try {
    $riff = [Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
    if ($riff -ne "RIFF") { throw "not a RIFF WAV: $Path" }
    [void]$reader.ReadUInt32()
    $wave = [Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
    if ($wave -ne "WAVE") { throw "not a WAVE file: $Path" }

    $sampleRate = 0
    $channels = 0
    $bitsPerSample = 0
    [byte[]]$audioBytes = @()

    while ($stream.Position + 8 -le $stream.Length) {
      $chunkId = [Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
      $chunkSize = [int]$reader.ReadUInt32()
      $chunkStart = $stream.Position
      if ($chunkId -eq "fmt ") {
        $format = $reader.ReadUInt16()
        $channels = $reader.ReadUInt16()
        $sampleRate = [int]$reader.ReadUInt32()
        [void]$reader.ReadUInt32()
        [void]$reader.ReadUInt16()
        $bitsPerSample = $reader.ReadUInt16()
        if ($format -ne 1 -or $bitsPerSample -ne 16) {
          throw "expected PCM16 WAV, got format=$format bits=$bitsPerSample"
        }
      } elseif ($chunkId -eq "data") {
        $audioBytes = $reader.ReadBytes($chunkSize)
      }
      $stream.Position = $chunkStart + $chunkSize + ($chunkSize % 2)
    }

    if ($sampleRate -le 0 -or $channels -le 0 -or $audioBytes.Count -eq 0) {
      throw "missing WAV fmt/data chunks: $Path"
    }

    $peak = 0.0
    $monoSamples = [Collections.Generic.List[double]]::new()
    $frameBytes = 2 * [Math]::Max(1, $channels)
    for ($offset = 0; $offset + ($frameBytes - 1) -lt $audioBytes.Count; $offset += $frameBytes) {
      $sum = 0.0
      for ($channel = 0; $channel -lt $channels; $channel++) {
        $sample = [BitConverter]::ToInt16($audioBytes, $offset + (2 * $channel))
        $value = [double]$sample / 32768.0
        if ([Math]::Abs($value) -gt $peak) { $peak = [Math]::Abs($value) }
        $sum += $value
      }
      $monoSamples.Add($sum / [double][Math]::Max(1, $channels))
    }

    $frameLen = [Math]::Max(1, [int][Math]::Round([double]$sampleRate * $FrameMs / 1000.0))
    $highZcrFrameCount = 0
    $maxZcr = 0.0
    $maxFrameRms = 0.0
    for ($start = 0; $start -lt $monoSamples.Count; $start += $frameLen) {
      $end = [Math]::Min($start + $frameLen, $monoSamples.Count)
      $count = [Math]::Max(1, $end - $start)
      $sumSq = 0.0
      $crossings = 0
      $prev = $monoSamples[$start]
      for ($i = $start; $i -lt $end; $i++) {
        $current = $monoSamples[$i]
        $sumSq += $current * $current
        if ($i -gt $start -and (($prev -lt 0.0 -and $current -ge 0.0) -or ($prev -ge 0.0 -and $current -lt 0.0))) {
          $crossings += 1
        }
        $prev = $current
      }
      $rms = [Math]::Sqrt($sumSq / [double]$count)
      $zcr = if ($count -gt 1) { [double]$crossings / [double]($count - 1) } else { 0.0 }
      if ($zcr -gt $maxZcr) { $maxZcr = $zcr }
      if ($rms -gt $maxFrameRms) { $maxFrameRms = $rms }
      if ($zcr -ge $HighZcrThreshold -and $rms -ge $HarshRmsThreshold) {
        $highZcrFrameCount += 1
      }
    }

    return [ordered]@{
      sample_rate = $sampleRate
      channels = $channels
      bits_per_sample = $bitsPerSample
      samples = [int64]($audioBytes.Count / 2 / [Math]::Max(1, $channels))
      duration_sec = [double]($audioBytes.Count / 2 / [Math]::Max(1, $channels)) / [double]$sampleRate
      peak = $peak
      high_zcr_frame_count = $highZcrFrameCount
      max_zcr = $maxZcr
      max_frame_rms = $maxFrameRms
    }
  } finally {
    $reader.Dispose()
    $stream.Dispose()
  }
}

function Invoke-AsrForCase([string]$WavPath, [string]$CaseRunId) {
  if (-not (Test-Path -LiteralPath $AsrRunner -PathType Leaf)) {
    throw "ASR runner not found: $AsrRunner"
  }
  $asrOut = New-Item -ItemType Directory -Force -Path (Join-Path $outRoot.FullName "asr\$CaseRunId")
  $asrLog = Join-Path $asrOut.FullName "asr_runner.log"
  & powershell -ExecutionPolicy Bypass -File $AsrRunner `
    -InputMedia $WavPath `
    -OutputDir $asrOut.FullName `
    -Engine $AsrEngine `
    -Model $AsrModel *> $asrLog
  if ($LASTEXITCODE -ne 0) {
    throw "ASR failed for $CaseRunId with exit code $LASTEXITCODE; see $asrLog"
  }
  return (Get-Content -LiteralPath (Join-Path $asrOut.FullName "transcript.txt") -Raw).Trim()
}

function Invoke-CargoChecked([string[]]$CargoArgs) {
  Write-Host ""
  Write-Host (">>> cargo " + ($CargoArgs -join " ")) -ForegroundColor Cyan
  & cargo @CargoArgs
  if ($LASTEXITCODE -ne 0) {
    throw "cargo command failed with exit code $LASTEXITCODE"
  }
}

$outRoot = New-Item -ItemType Directory -Force -Path $OutDir
$promptSpec = Read-JsonFile -Path (Resolve-Path -LiteralPath $PromptSet).Path
$cases = [Collections.Generic.List[object]]::new()
foreach ($case in $promptSpec.cases) {
  if ($CaseIds.Count -eq 0 -or $CaseIds -contains $case.id) {
    $cases.Add([ordered]@{
      id = $case.id
      text = $case.text
      required_terms = @($case.required_terms)
      is_clone = $false
    })
  }
}
if (-not $SkipClone -and -not [string]::IsNullOrWhiteSpace($RefAudio) -and ($CaseIds.Count -eq 0 -or $CaseIds -contains "clone")) {
  $cases.Add([ordered]@{
    id = "clone"
    text = "这是声音复制的噪声回归测试。"
    required_terms = @("声音复制", "噪声回归")
    is_clone = $true
  })
}
if ($cases.Count -eq 0) {
  throw "no cases selected"
}

$featureArgs = Get-CargoFeatureArgs -DeviceName $Device
$rows = [Collections.Generic.List[object]]::new()
$comboIndex = 0

foreach ($seed in $Seeds) {
  foreach ($cfg in $CfgValues) {
    foreach ($latentNorm in $LatentNormValues) {
      foreach ($scheduler in $Schedulers) {
        $comboIndex += 1
        if ($MaxCombos -gt 0 -and $comboIndex -gt $MaxCombos) {
          break
        }

        $cfgId = "seed${seed}_cfg$(ConvertTo-SafeId "$cfg")_ln$(ConvertTo-SafeId $latentNorm)_$scheduler"
        Write-Host ""
        Write-Host "=== Sweep $cfgId ===" -ForegroundColor Magenta

        foreach ($case in $cases) {
          $caseRunId = "$($case.id)_$cfgId"
          $wavPath = Join-Path $outRoot.FullName "$caseRunId.wav"
          $metricsPath = Join-Path $outRoot.FullName "$caseRunId.metrics.json"

          $commandName = if ($case.is_clone) { "clone" } else { "synth" }
          $args = @("run", "-p", "voxcpm2-cli") + $featureArgs + @(
            "--", $commandName,
            "--model-dir", $ModelDir,
            "--device", $Device,
            "--text", $case.text,
            "--out", $wavPath,
            "--metrics-out", $metricsPath,
            "--cfg", "$cfg",
            "--steps", "$Steps",
            "--seed", "$seed",
            "--t-scheduler", $scheduler
          )
          if (-not [string]::IsNullOrWhiteSpace($latentNorm)) {
            $args += @("--latent-norm", $latentNorm)
          }
          if ($case.is_clone) {
            $args += @("--ref-audio", $RefAudio, "--i-have-consent")
          }
          if ($DryRun) {
            $args += "--dry-run"
          }

          Invoke-CargoChecked -CargoArgs $args
          $metrics = Read-JsonFile -Path $metricsPath
          $quality = Read-Pcm16WavMetrics -Path $wavPath -FrameMs 20.0 -HighZcrThreshold 0.11 -HarshRmsThreshold 0.005
          $transcript = ""
          $similarity = $null
          $termsPassed = $false
          $missingTerms = @()
          if ($RunAsr) {
            $transcript = Invoke-AsrForCase -WavPath $wavPath -CaseRunId $caseRunId
            $similarity = Get-TextSimilarity -Expected $case.text -Actual $transcript
            foreach ($term in $case.required_terms) {
              if (-not $transcript.Contains($term)) {
                $missingTerms += $term
              }
            }
            $termsPassed = ($missingTerms.Count -eq 0)
          }

          $quietAfter = if ($null -ne $metrics.polish) { [double]$metrics.polish.quiet_rms_after } else { $null }
          $cloneQuietAfter = if ($null -ne $metrics.clone_reference_polish) { [double]$metrics.clone_reference_polish.quiet_rms_after } else { $null }
          $score = 0.0
          if ($RunAsr -and $null -ne $similarity) { $score += 100.0 * [double]$similarity }
          if ($RunAsr -and $termsPassed) { $score += 25.0 }
          $score -= [double]$quality.high_zcr_frame_count * 0.4
          if ($null -ne $quietAfter) { $score -= [double]$quietAfter * 1000.0 }
          if ($null -ne $cloneQuietAfter) { $score -= [double]$cloneQuietAfter * 300.0 }
          if ([double]$quality.peak -gt 0.90) { $score -= 20.0 }

          $row = [pscustomobject]@{
            config_id = $cfgId
            case_id = $case.id
            is_clone = [bool]$case.is_clone
            seed = $seed
            cfg = $cfg
            latent_norm = $latentNorm
            scheduler = $scheduler
            wav_path = ConvertTo-RelativePath -Path $wavPath
            metrics_path = ConvertTo-RelativePath -Path $metricsPath
            sample_rate = $quality.sample_rate
            duration_sec = $quality.duration_sec
            peak = $quality.peak
            high_zcr_frame_count = $quality.high_zcr_frame_count
            max_zcr = $quality.max_zcr
            quiet_rms_after = $quietAfter
            clone_ref_quiet_rms_after = $cloneQuietAfter
            asr_terms_passed = if ($RunAsr) { $termsPassed } else { $null }
            asr_similarity = $similarity
            asr_missing_terms = ($missingTerms -join ",")
            asr_transcript = $transcript
            score = $score
          }
          $rows.Add($row)
        }
      }
      if ($MaxCombos -gt 0 -and $comboIndex -gt $MaxCombos) { break }
    }
    if ($MaxCombos -gt 0 -and $comboIndex -gt $MaxCombos) { break }
  }
  if ($MaxCombos -gt 0 -and $comboIndex -gt $MaxCombos) { break }
}

$csvPath = Join-Path $outRoot.FullName "quality_sweep_results.csv"
$jsonPath = Join-Path $outRoot.FullName "quality_sweep_results.json"
$summaryPath = Join-Path $outRoot.FullName "quality_sweep_summary.md"
$rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8
$rows | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$configSummary = $rows |
  Group-Object config_id |
  ForEach-Object {
    $group = $_.Group
    [pscustomobject]@{
      config_id = $_.Name
      cases = $group.Count
      terms_passed = @($group | Where-Object { $_.asr_terms_passed -eq $true }).Count
      avg_similarity = if ($RunAsr) { ($group | Measure-Object asr_similarity -Average).Average } else { $null }
      total_high_zcr = ($group | Measure-Object high_zcr_frame_count -Sum).Sum
      max_peak = ($group | Measure-Object peak -Maximum).Maximum
      avg_score = ($group | Measure-Object score -Average).Average
    }
  } |
  Sort-Object @{ Expression = "terms_passed"; Descending = $true }, @{ Expression = "avg_score"; Descending = $true }

$summaryLines = [Collections.Generic.List[string]]::new()
$summaryLines.Add("# VoxCPM2 Quality Sweep")
$summaryLines.Add("")
$summaryLines.Add("- generated_at: $((Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"))")
$summaryLines.Add("- device: $Device")
$summaryLines.Add("- dry_run: $([bool]$DryRun)")
$summaryLines.Add("- asr_enabled: $([bool]$RunAsr)")
$summaryLines.Add("- results_csv: $(ConvertTo-RelativePath -Path $csvPath)")
$summaryLines.Add("- results_json: $(ConvertTo-RelativePath -Path $jsonPath)")
$summaryLines.Add("")
$summaryLines.Add("## Top Configs")
$summaryLines.Add("")
$summaryLines.Add("| config | cases | terms_passed | avg_similarity | total_high_zcr | max_peak | avg_score |")
$summaryLines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in @($configSummary | Select-Object -First 10)) {
  $simText = if ($null -eq $item.avg_similarity) { "n/a" } else { "{0:n4}" -f [double]$item.avg_similarity }
  $summaryLines.Add(("| {0} | {1} | {2} | {3} | {4} | {5:n4} | {6:n2} |" -f $item.config_id, $item.cases, $item.terms_passed, $simText, $item.total_high_zcr, [double]$item.max_peak, [double]$item.avg_score))
}
$summaryLines.Add("")
$summaryLines.Add("## Notes")
$summaryLines.Add("")
$summaryLines.Add("- Score favors ASR text similarity and required-term pass, then penalizes high-ZCR frames, quiet RMS, clone reference floor, and near-clipping.")
$summaryLines.Add("- Use a small `-MaxCombos` probe first, then run the full matrix on CUDA when you are ready to spend GPU time.")
($summaryLines -join "`r`n") + "`r`n" | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "Quality sweep completed." -ForegroundColor Green
Write-Host "CSV:     $csvPath" -ForegroundColor Green
Write-Host "JSON:    $jsonPath" -ForegroundColor Green
Write-Host "Summary: $summaryPath" -ForegroundColor Green
