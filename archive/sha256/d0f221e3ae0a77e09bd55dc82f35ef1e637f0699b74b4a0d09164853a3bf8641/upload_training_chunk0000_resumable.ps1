[CmdletBinding()]
param(
    [switch]$ProbeOnly,
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'
$archivePath = 'E:\Lighten_VLA_data_staging\physicalai_train_chunk0000_20261002.tar'
$remoteDirectory = '/home/tuqiang/lang/Lighten_VLA/incoming/'
$remoteArchive = $remoteDirectory + [IO.Path]::GetFileName($archivePath)
$toolDirectory = Join-Path $PSScriptRoot 'tools\winscp-6.5.7'
$assembly = Join-Path $toolDirectory 'lib\net40\WinSCPnet.dll'
if ($PSVersionTable.PSEdition -eq 'Core') {
    $assembly = Join-Path $toolDirectory 'lib\netstandard2.0\WinSCPnet.dll'
}
$executable = Join-Path $toolDirectory 'tools\WinSCP.exe'
if (-not (Test-Path -LiteralPath $assembly) -or -not (Test-Path -LiteralPath $executable)) {
    throw 'The prepared WinSCP 6.5.7 tools are missing.'
}
if ((Get-AuthenticodeSignature -LiteralPath $executable).Status -ne 'Valid') {
    throw 'WinSCP executable signature verification failed.'
}
Add-Type -Path $assembly

# Pin the existing, previously verified OpenSSH host key; never accept an arbitrary key.
$hostKey = 'AAAAC3NzaC1lZDI1NTE5AAAAIGMgeYKiYWtrBuPrHYIYXiZ8oCSEmoL/GQMxF9rekx+Y'
$sha = [Security.Cryptography.SHA256]::Create()
try { $fingerprint = [Convert]::ToBase64String($sha.ComputeHash([Convert]::FromBase64String($hostKey))).TrimEnd('=') }
finally { $sha.Dispose() }
$options = New-Object WinSCP.SessionOptions -Property @{
    Protocol = [WinSCP.Protocol]::Sftp
    HostName = '127.0.0.1'
    PortNumber = 2222
    UserName = 'tuqiang'
    SshHostKeyFingerprint = ('ssh-ed25519 255 ' + $fingerprint)
    Timeout = [TimeSpan]::FromSeconds(45)
}
$options.AddRawSettings('SendBuf', '262144')
$options.AddRawSettings('PingType', '1')
$options.AddRawSettings('PingIntervalSecs', '15')

if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) { throw 'Download archive is missing.' }
$archiveBytes = (Get-Item -LiteralPath $archivePath).Length
$logDirectory = Join-Path $PSScriptRoot ('results\transfer_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$script:progressPath = Join-Path $logDirectory 'progress.json'
$script:lastProgressAt = [DateTime]::MinValue

if (-not $ProbeOnly -and -not $SelfTest) {
    $oldScp = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'scp.exe' -and $_.CommandLine -match 'physicalai_train_chunk0000_20261002'
    })
    if ($oldScp.Count -gt 0) { throw 'Stop the previous SCP upload with Ctrl+C first; keep the server partial file.' }
    $checksumFile = $archivePath + '.sha256'
    if (-not (Test-Path -LiteralPath $checksumFile)) { throw 'Archive checksum file is missing.' }
    $expected = ([IO.File]::ReadAllText($checksumFile) -split '\s+')[0].ToLowerInvariant()
    if ($expected -notmatch '^[a-f0-9]{64}$') { throw 'Invalid archive checksum.' }
    Write-Host 'Checking local archive SHA256...'
    if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw 'Local archive SHA256 mismatch.'
    }
}
# Prompt once. Password is kept in memory, not in scripts, process arguments or logs.
$options.SecurePassword = Read-Host 'SSH password for tuqiang' -AsSecureString
$transfer = New-Object WinSCP.TransferOptions
$transfer.TransferMode = [WinSCP.TransferMode]::Binary
$transfer.OverwriteMode = [WinSCP.OverwriteMode]::Resume
# Resume the existing SCP destination directly, rather than starting a separate .filepart.
$transfer.ResumeSupport.State = [WinSCP.TransferResumeSupportState]::Off
$transfer.SpeedLimit = 0

$uploadLock = $null
try {
    if (-not $ProbeOnly -and -not $SelfTest) {
        $uploadLock = [IO.File]::Open(($archivePath + '.upload.lock'), [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    }
    $attempt = 0
    while ($true) {
        $attempt++
        $session = New-Object WinSCP.Session
        $session.ExecutablePath = $executable
        $session.ReconnectTime = [TimeSpan]::FromMinutes(10)
        $session.SessionLogPath = Join-Path $logDirectory ('attempt_{0:D3}.log' -f $attempt)
        $session.add_FileTransferProgress({
            param($sender, $eventArgs)
            $e = $eventArgs
            if ($null -eq $e) { $e = $_ }
            if ($null -eq $e -or $e -isnot [WinSCP.FileTransferProgressEventArgs]) { return }
            $now = [DateTime]::UtcNow
            if (($now - $script:lastProgressAt).TotalSeconds -ge 5) {
                $script:lastProgressAt = $now
                $progress = [ordered]@{ utc=$now.ToString('o'); file=$e.FileName;
                    percent=[Math]::Round(100*$e.FileProgress,2); bytes_per_second=$e.CPS }
                [IO.File]::WriteAllText($script:progressPath, ($progress | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
                Write-Host ('{0:F1}% | {1:F2} MiB/s' -f (100*$e.FileProgress), ($e.CPS/1MB))
            }
        })
        try {
            Write-Host ('Connecting, attempt {0}; logs: {1}' -f $attempt, $logDirectory)
            $session.Open($options)
            if ($SelfTest) {
                # Test resuming an existing destination on this actual SFTP server.
                $testId = 'resume_test_' + [Guid]::NewGuid().ToString('N')
                $localTest = Join-Path $logDirectory ($testId + '.bin')
                $bytes = New-Object byte[] (1MB)
                $random = [Security.Cryptography.RandomNumberGenerator]::Create()
                try { $random.GetBytes($bytes) } finally { $random.Dispose() }
                [IO.File]::WriteAllBytes($localTest, $bytes[0..65535])
                $remoteTest = $remoteDirectory + $testId + '.bin'
                $session.PutFiles($localTest, $remoteTest, $false, $transfer).Check()
                if ($session.GetFileInfo($remoteTest).Length -ne 65536) { throw [IO.InvalidDataException]::new('Test partial size mismatch.') }
                [IO.File]::WriteAllBytes($localTest, $bytes)
                $session.PutFiles($localTest, $remoteTest, $false, $transfer).Check()
                $check = $session.ExecuteCommand("sha256sum '$remoteTest'")
                $check.Check()
                $testHash = (Get-FileHash -LiteralPath $localTest -Algorithm SHA256).Hash.ToLowerInvariant()
                if (($check.Output -split '\s+')[0] -ne $testHash) { throw [IO.InvalidDataException]::new('Resume test SHA256 mismatch.') }
                Write-Host 'SELFTEST VERIFIED: resumed existing 64 KiB destination to 1 MiB; SHA256 matched.'
                break
            }
            if (-not $session.FileExists($remoteDirectory)) { $session.CreateDirectory($remoteDirectory) }
            $remoteBytes = 0
            if ($session.FileExists($remoteArchive)) { $remoteBytes = $session.GetFileInfo($remoteArchive).Length }
            if ($remoteBytes -gt $archiveBytes) { throw [IO.InvalidDataException]::new('Remote file is larger than local archive; refusing to overwrite.') }
            Write-Host ('Existing server file: {0:N0} / {1:N0} bytes ({2:F2}%)' -f $remoteBytes,$archiveBytes,(100*$remoteBytes/$archiveBytes))
            if ($ProbeOnly) { Write-Host 'PROBE VERIFIED: pinned host key, authentication, SFTP and remote file size.'; break }
            if ($remoteBytes -lt $archiveBytes) {
                $session.PutFiles($archivePath, $remoteArchive, $false, $transfer).Check()
            }
            if ($session.GetFileInfo($remoteArchive).Length -ne $archiveBytes) { throw [IO.InvalidDataException]::new('Remote archive size mismatch.') }
            $checksumTransfer = New-Object WinSCP.TransferOptions
            $checksumTransfer.TransferMode = [WinSCP.TransferMode]::Binary
            $checksumTransfer.ResumeSupport.State = [WinSCP.TransferResumeSupportState]::Off
            $session.PutFiles($checksumFile, ($remoteArchive + '.sha256'), $false, $checksumTransfer).Check()
            Write-Host 'Checking server archive SHA256...'
            $result = $session.ExecuteCommand("cd '$remoteDirectory' && sha256sum -c 'physicalai_train_chunk0000_20261002.tar.sha256'")
            if ($result.ExitCode -ne 0) { throw [IO.InvalidDataException]::new('Server SHA256 failed. Partial file retained; do not train from it.') }
            [IO.File]::WriteAllText((Join-Path $logDirectory 'verified.json'),
                ([ordered]@{status='verified';sha256=$expected;bytes=$archiveBytes;attempts=$attempt;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json),
                [Text.UTF8Encoding]::new($false))
            Write-Host $result.Output
            Write-Host 'UPLOAD VERIFIED. Reply in Codex: 下载上传完成'
            break
        }
        catch {
            $cause = $_.Exception
            while ($cause.InnerException) { $cause = $cause.InnerException }
            if ($ProbeOnly -or $SelfTest -or $cause -is [IO.InvalidDataException] -or
                $cause.Message -match 'Authentication failed|Access denied|Host key does not match|Permission denied|No space left') { throw }
            Write-Warning ('Connection/transfer interrupted: {0}. Retrying in 10 seconds from retained bytes; Ctrl+C stops safely.' -f $cause.Message)
        }
        finally { $session.Dispose() }
        Start-Sleep -Seconds 10
    }
}
finally {
    if ($uploadLock) { $uploadLock.Dispose() }
    $options.SecurePassword.Dispose()
}
