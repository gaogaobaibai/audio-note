<#
.SYNOPSIS
    一键安装 audio_note：uv 创建 Python 3.12 虚拟环境 → 安装依赖 → 下载模型 → 生成 config.yaml。

.DESCRIPTION
    需要先安装 uv（https://docs.astral.sh/uv/）：
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    或 winget install astral-sh.uv

.PARAMETER SkipModels
    跳过模型下载（例如已手动放好 models/）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/install.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "== audio_note 安装 ($ProjectRoot) =="

# 1. uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 uv。请先安装："
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host "  或 winget install astral-sh.uv"
    exit 1
}

# 2. 虚拟环境
if (-not (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) {
    Write-Host "[1/4] 创建虚拟环境 .venv (Python 3.12)"
    uv venv --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv venv 失败" }
} else {
    Write-Host "[1/4] .venv 已存在，跳过创建"
}

# 3. 依赖
Write-Host "[2/4] 安装依赖 (uv pip install -e .)"
uv pip install -e .
if ($LASTEXITCODE -ne 0) { throw "uv pip install 失败" }
Write-Host "      开发依赖 pytest"
uv pip install pytest
if ($LASTEXITCODE -ne 0) { throw "安装 pytest 失败" }

# 4. 模型
if ($SkipModels) {
    Write-Host "[3/4] 跳过模型下载 (-SkipModels)"
} else {
    Write-Host "[3/4] 下载模型"
    & (Join-Path $PSScriptRoot "download_models.ps1")
}

# 5. 配置
Write-Host "[4/4] 配置文件"
$cfg = Join-Path $ProjectRoot "config.yaml"
if (-not (Test-Path $cfg)) {
    Copy-Item (Join-Path $ProjectRoot "config.example.yaml") $cfg
    Write-Host "      已生成 config.yaml，请填写 mimo.api_key（或设置环境变量 MIMO_API_KEY）"
} else {
    Write-Host "      config.yaml 已存在，保持不变"
}
$gl = Join-Path $ProjectRoot "glossary.txt"
if (-not (Test-Path $gl)) {
    Copy-Item (Join-Path $ProjectRoot "glossary.example.txt") $gl
    Write-Host "      已生成 glossary.txt（纠错词表，可按需编辑）"
}

Write-Host ""
Write-Host "安装完成。下一步："
Write-Host "  .\.venv\Scripts\audio-note.exe init        # 检查配置与模型"
Write-Host "  .\.venv\Scripts\audio-note.exe devices     # 查看输入设备"
Write-Host "  .\.venv\Scripts\audio-note.exe run         # 开始录音 + 自动处理"
Write-Host "  （先在 PowerShell 里 .\.venv\Scripts\Activate.ps1 之后可以直接用 audio-note）"
