<#
.SYNOPSIS
    把 audio_note 注册为 Windows 计划任务：用户登录时后台静默启动 `pythonw.exe -m audio_note run`。

.DESCRIPTION
    使用 schtasks /create /sc onlogon。pythonw.exe 无控制台窗口；日志见 data/audio_note.log。
    用 -Remove 删除任务；用 -Tray 启动时带托盘图标。

.PARAMETER TaskName
    计划任务名称，默认 AudioNote。

.PARAMETER Tray
    传 --tray 给 run 命令（系统托盘图标）。

.PARAMETER Remove
    删除已注册的计划任务。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/register_autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts/register_autostart.ps1 -Tray
    powershell -ExecutionPolicy Bypass -File scripts/register_autostart.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$TaskName = "AudioNote",
    [switch]$Tray,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

if ($Remove) {
    schtasks /delete /tn $TaskName /f
    if ($LASTEXITCODE -ne 0) { throw "删除计划任务失败 (exit $LASTEXITCODE)" }
    Write-Host "已删除计划任务 $TaskName"
    exit 0
}

$pythonw = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    throw "未找到 $pythonw ，请先运行 scripts/install.ps1"
}
if (-not (Test-Path (Join-Path $ProjectRoot "config.yaml"))) {
    Write-Warning "config.yaml 不存在，服务启动后会使用默认配置且缺少 API key；请先运行 audio-note init"
}

# config.yaml / data / models 都按项目根目录解析，所以不依赖工作目录。
$args_ = "-m audio_note run"
if ($Tray) { $args_ += " --tray" }
# schtasks 的 /tr 里：整条命令用双引号包起来，内部路径再用 \" 转义
$tr = "\`"$pythonw\`" $args_"

Write-Host "注册计划任务 $TaskName :"
Write-Host "  $pythonw $args_"
schtasks /create /tn $TaskName /sc onlogon /rl limited /f /tr $tr
if ($LASTEXITCODE -ne 0) { throw "schtasks /create 失败 (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "完成。下次登录时自动启动；现在立即启动："
Write-Host "  schtasks /run /tn $TaskName"
Write-Host "查看状态 / 停止："
Write-Host "  schtasks /query /tn $TaskName"
Write-Host "  schtasks /end /tn $TaskName"
Write-Host "提示：笔记本用电池时若不启动，可在 任务计划程序 中编辑该任务的 条件 页，取消 [只有在计算机使用交流电源时才启动]。"
