<#
.SYNOPSIS
    下载 audio_note 需要的三个本地模型到 models/ 目录（已存在则跳过）。

.DESCRIPTION
    1. silero_vad.onnx                                        (Silero VAD v4, ~0.6 MB)
    2. 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx     (3D-Speaker CAM++ 声纹, ~27 MB)
    3. sherpa-onnx-pyannote-segmentation-3-0/model.onnx       (pyannote segmentation-3.0, tar.bz2 解压, ~6 MB)
    模型均来自 k2-fsa/sherpa-onnx 的 GitHub Releases。

.PARAMETER ModelsDir
    目标目录，默认为项目根目录下的 models。

.PARAMETER Force
    即使文件已存在也重新下载。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
#>
[CmdletBinding()]
param(
    [string]$ModelsDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ModelsDir) { $ModelsDir = Join-Path $ProjectRoot "models" }
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# 让 Invoke-WebRequest 快一些（关闭进度条）并支持 TLS 1.2
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Models = @(
    @{
        Name = "silero_vad.onnx"
        Url  = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
        Dest = "silero_vad.onnx"
    },
    @{
        Name = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
        Url  = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
        Dest = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    },
    @{
        Name    = "sherpa-onnx-pyannote-segmentation-3-0"
        Url     = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        Dest    = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
        Archive = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    }
)

function Download-File([string]$Url, [string]$Target) {
    $tmp = "$Target.part"
    Write-Host "  下载 $Url"
    Invoke-WebRequest -Uri $Url -OutFile $tmp -UseBasicParsing
    Move-Item -Force $tmp $Target
}

foreach ($m in $Models) {
    $dest = Join-Path $ModelsDir $m.Dest
    if ((Test-Path $dest) -and -not $Force) {
        Write-Host "[跳过] $($m.Name) 已存在: $dest"
        continue
    }
    Write-Host "[下载] $($m.Name)"
    if ($m.Archive) {
        $archive = Join-Path $ModelsDir $m.Archive
        Download-File $m.Url $archive
        # Windows 10 1803+ 自带 bsdtar，可直接解 tar.bz2
        $tar = Get-Command tar -ErrorAction SilentlyContinue
        if (-not $tar) { throw "未找到 tar 命令，无法解压 $archive；请手动解压到 $ModelsDir" }
        Write-Host "  解压 $archive"
        & tar -xjf $archive -C $ModelsDir
        if ($LASTEXITCODE -ne 0) { throw "tar 解压失败 (exit $LASTEXITCODE): $archive" }
        Remove-Item -Force $archive
    } else {
        Download-File $m.Url $dest
    }
    if (-not (Test-Path $dest)) { throw "下载后未找到文件: $dest" }
    $size = [math]::Round((Get-Item $dest).Length / 1MB, 1)
    Write-Host "  完成: $dest ($size MB)"
}

Write-Host ""
Write-Host "模型就绪: $ModelsDir"
