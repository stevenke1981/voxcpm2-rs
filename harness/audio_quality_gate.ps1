param(
  [string]$ModelDir = "models/VoxCPM2",
  [string]$Device = "cuda",
  [string]$OutDir = "output/quality-gate",
  [string]$PromptSet = "tests/golden/mandarin_regression_prompts.json",
  [string]$RefAudio = "",
  [int]$Steps = 30,
  [int]$Seed = 99,
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
    for ($i = 0; $i + 1 -lt $audioBytes.Count; $i += 2) {
      $sample = [BitConverter]::ToInt16($audioBytes, $i)
      $value = [Math]::Abs([double]$sample / 32768.0)
      if ($value -gt $peak) { $peak = $value }
    }

    [pscustomobject]@{
      sample_rate = $sampleRate
      channels = $channels
      bits_per_sample = $bitsPerSample
      samples = [int64]($audioBytes.Count / 2 / [Math]::Max(1, $channels))
      duration_sec = [double]($audioBytes.Count / 2 / [Math]::Max(1, $channels)) / [double]$sampleRate
      peak = $peak
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

  Write-Host ("validated {0}: {1} Hz, {2:n2}s, peak={3:n4}" -f $WavPath, $wav.sample_rate, $wav.duration_sec, $wav.peak) -ForegroundColor Green
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
Write-Host "Audio quality gate completed. ASR transcript comparison is the next acceptance layer for real model outputs." -ForegroundColor Green
