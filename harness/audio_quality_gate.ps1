param(
  [string]$ModelDir = "models/VoxCPM2",
  [string]$Device = "cuda",
  [string]$OutDir = "output/quality-gate",
  [string]$PromptSet = "tests/golden/mandarin_regression_prompts.json",
  [string]$RefAudio = "",
  [int]$Steps = 30,
  [int]$Seed = 99,
  [double]$FrameMs = 20.0,
  [double]$HighZcrThreshold = 0.11,
  [double]$HarshRmsThreshold = 0.005,
  [switch]$DryRun,
  [switch]$SkipClone
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-CargoFeatureArgs([string]$DeviceName) {
  switch ($DeviceName.ToLowerInvariant()) {
    "cuda" { return @("--no-default-features", "--features", "cuda") }
    "metal" { return @("--no-default-features", "--features", "metal") }
    "mkl" { return @("--no-default-features", "--features", "mkl") }
    default { return @("--features", "cpu") }
  }
}

function Invoke-Checked([string[]]$CargoArgs) {
  Write-Host ""
  Write-Host (">>> cargo " + ($CargoArgs -join " ")) -ForegroundColor Cyan
  & cargo @CargoArgs
  if ($LASTEXITCODE -ne 0) {
    throw "cargo command failed with exit code $LASTEXITCODE"
  }
}

function Read-Pcm16WavMetrics([string]$Path) {
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
        $abs = [Math]::Abs($value)
        if ($abs -gt $peak) { $peak = $abs }
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
        if ($i -gt $start) {
          if (($prev -lt 0.0 -and $current -ge 0.0) -or ($prev -ge 0.0 -and $current -lt 0.0)) {
            $crossings += 1
          }
        }
        $prev = $current
      }

      $rms = [Math]::Sqrt($sumSq / [double]$count)
      $zcr = 0.0
      if ($count -gt 1) {
        $zcr = [double]$crossings / [double]($count - 1)
      }
      if ($zcr -gt $maxZcr) { $maxZcr = $zcr }
      if ($rms -gt $maxFrameRms) { $maxFrameRms = $rms }
      if ($zcr -ge $HighZcrThreshold -and $rms -ge $HarshRmsThreshold) {
        $highZcrFrameCount += 1
      }
    }

    [pscustomobject]@{
      sample_rate = $sampleRate
      channels = $channels
      bits_per_sample = $bitsPerSample
      samples = [int64]($audioBytes.Count / 2 / [Math]::Max(1, $channels))
      duration_sec = [double]($audioBytes.Count / 2 / [Math]::Max(1, $channels)) / [double]$sampleRate
      peak = $peak
      frame_ms = $FrameMs
      high_zcr_threshold = $HighZcrThreshold
      harsh_rms_threshold = $HarshRmsThreshold
      high_zcr_frame_count = $highZcrFrameCount
      max_zcr = $maxZcr
      max_frame_rms = $maxFrameRms
    }
  } finally {
    $reader.Dispose()
    $stream.Dispose()
  }
}

function Assert-Metrics([string]$MetricsPath, [string]$WavPath, [bool]$ExpectedClone, [bool]$ExpectedDryRun) {
  if (-not (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
    throw "missing metrics JSON: $MetricsPath"
  }
  if (-not (Test-Path -LiteralPath $WavPath -PathType Leaf)) {
    throw "missing WAV: $WavPath"
  }

  $json = Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json
  $wav = Read-Pcm16WavMetrics -Path $WavPath

  if ([bool]$json.is_clone -ne $ExpectedClone) {
    throw "metrics is_clone mismatch for $MetricsPath"
  }
  if ([bool]$json.dry_run -ne $ExpectedDryRun) {
    throw "metrics dry_run mismatch for $MetricsPath"
  }
  if ([int64]$json.samples -le 0 -or $wav.samples -le 0) {
    throw "empty audio in $WavPath"
  }
  if ([int]$json.sample_rate -ne $wav.sample_rate) {
    throw "sample-rate mismatch: metrics=$($json.sample_rate) wav=$($wav.sample_rate)"
  }
  if ($wav.channels -ne 1) {
    throw "expected mono WAV, got $($wav.channels) channels"
  }
  if ($wav.peak -gt 0.951) {
    throw "WAV peak exceeds headroom gate: $($wav.peak)"
  }
  if (-not $ExpectedDryRun -and $null -eq $json.polish) {
    throw "real synthesis metrics must include output polish report"
  }
  if ($ExpectedClone -and -not $ExpectedDryRun -and $null -eq $json.clone_reference_polish) {
    throw "real clone metrics must include clone_reference_polish"
  }

  $qualityPath = [IO.Path]::ChangeExtension($MetricsPath, ".quality.json")
  [ordered]@{
    wav_path = (Resolve-Path -LiteralPath $WavPath).Path
    metrics_path = (Resolve-Path -LiteralPath $MetricsPath).Path
    sample_rate = $wav.sample_rate
    channels = $wav.channels
    bits_per_sample = $wav.bits_per_sample
    samples = $wav.samples
    duration_sec = $wav.duration_sec
    peak = $wav.peak
    frame_ms = $wav.frame_ms
    high_zcr_threshold = $wav.high_zcr_threshold
    harsh_rms_threshold = $wav.harsh_rms_threshold
    high_zcr_frame_count = $wav.high_zcr_frame_count
    max_zcr = $wav.max_zcr
    max_frame_rms = $wav.max_frame_rms
  } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $qualityPath -Encoding UTF8

  Write-Host ("validated {0}: {1} Hz, {2:n2}s, peak={3:n4}, high_zcr_frames={4}, max_zcr={5:n4}" -f $WavPath, $wav.sample_rate, $wav.duration_sec, $wav.peak, $wav.high_zcr_frame_count, $wav.max_zcr) -ForegroundColor Green
}

$promptPath = Resolve-Path -LiteralPath $PromptSet
$promptSpec = Get-Content -LiteralPath $promptPath -Raw | ConvertFrom-Json
$outRoot = New-Item -ItemType Directory -Force -Path $OutDir
$featureArgs = Get-CargoFeatureArgs -DeviceName $Device

foreach ($case in $promptSpec.cases) {
  $wavPath = Join-Path $outRoot.FullName "$($case.id).wav"
  $metricsPath = Join-Path $outRoot.FullName "$($case.id).metrics.json"
  $args = @(
    "run", "-p", "voxcpm2-cli"
  ) + $featureArgs + @(
    "--", "synth",
    "--model-dir", $ModelDir,
    "--device", $Device,
    "--text", $case.text,
    "--out", $wavPath,
    "--metrics-out", $metricsPath,
    "--steps", "$Steps",
    "--seed", "$Seed"
  )
  if ($DryRun) { $args += "--dry-run" }
  Invoke-Checked -CargoArgs $args
  Assert-Metrics -MetricsPath $metricsPath -WavPath $wavPath -ExpectedClone $false -ExpectedDryRun ([bool]$DryRun)
}

if (-not $SkipClone) {
  if ([string]::IsNullOrWhiteSpace($RefAudio)) {
    Write-Host "Clone gate skipped: pass -RefAudio or use -SkipClone." -ForegroundColor Yellow
  } else {
    $cloneWav = Join-Path $outRoot.FullName "clone_noisy_ref_seed99.wav"
    $cloneMetrics = Join-Path $outRoot.FullName "clone_noisy_ref_seed99.metrics.json"
    $cloneArgs = @(
      "run", "-p", "voxcpm2-cli"
    ) + $featureArgs + @(
      "--", "clone",
      "--model-dir", $ModelDir,
      "--device", $Device,
      "--ref-audio", $RefAudio,
      "--text", "这是声音复制的噪声回归测试。",
      "--i-have-consent",
      "--out", $cloneWav,
      "--metrics-out", $cloneMetrics,
      "--steps", "$Steps",
      "--seed", "$Seed"
    )
    if ($DryRun) { $cloneArgs += "--dry-run" }
    Invoke-Checked -CargoArgs $cloneArgs
    Assert-Metrics -MetricsPath $cloneMetrics -WavPath $cloneWav -ExpectedClone $true -ExpectedDryRun ([bool]$DryRun)
  }
}

Write-Host ""
Write-Host "Audio quality gate completed. Frame-level quality sidecars are ready; ASR transcript comparison is the next acceptance layer for real model outputs." -ForegroundColor Green
