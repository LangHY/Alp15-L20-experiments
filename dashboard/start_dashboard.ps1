$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$runtimePython = 'C:\Users\LANG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) { $runtimePython = (Get-Command python).Source }
$logFolder = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $logFolder -Force | Out-Null
try {
    Invoke-RestMethod 'http://127.0.0.1:8765/api/snapshot' -TimeoutSec 2 | Out-Null
    Write-Host 'Dashboard already available: http://127.0.0.1:8765'
} catch {
    $arguments = '"{0}" --root "{1}" --port 8765 --remote http://127.0.0.1:18766' -f (Join-Path $PSScriptRoot 'server.py'), $projectRoot
    $process = Start-Process -FilePath $runtimePython -ArgumentList $arguments -WindowStyle Hidden -PassThru -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $logFolder 'server.stdout.log') -RedirectStandardError (Join-Path $logFolder 'server.stderr.log')
    $process.Id | Set-Content (Join-Path $logFolder 'server.pid')
    Write-Host 'Dashboard started: http://127.0.0.1:8765'
}
