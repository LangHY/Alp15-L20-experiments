$ErrorActionPreference = 'Stop'
$archivePath = 'E:\Lighten_VLA_data_staging\physicalai_train_chunk0000_20261002.tar'
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) { throw 'Run the download script first.' }
if (-not (Test-Path -LiteralPath ($archivePath + '.sha256') -PathType Leaf)) { throw 'Archive checksum file is missing.' }
$expected = ([IO.File]::ReadAllText($archivePath + '.sha256') -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'Local archive SHA256 mismatch.' }
$sshOptions = @('-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','PubkeyAuthentication=no','-o','PreferredAuthentications=password')
& ssh -p 2222 @sshOptions 'tuqiang@127.0.0.1' 'mkdir -p /home/tuqiang/lang/Lighten_VLA/incoming'
if ($LASTEXITCODE -ne 0) { throw 'SSH connection failed. Restore the 2222 port mapping first.' }
& scp -P 2222 @sshOptions $archivePath ($archivePath + '.sha256') 'tuqiang@127.0.0.1:/home/tuqiang/lang/Lighten_VLA/incoming/'
if ($LASTEXITCODE -ne 0) { throw 'Transfer failed. The server partial file is retained; use sftp reput to resume it.' }
& ssh -p 2222 @sshOptions 'tuqiang@127.0.0.1' 'cd /home/tuqiang/lang/Lighten_VLA/incoming && sha256sum -c physicalai_train_chunk0000_20261002.tar.sha256'
if ($LASTEXITCODE -ne 0) { throw 'Server archive SHA256 verification failed.' }
Write-Host 'UPLOAD VERIFIED. Reply in Codex: 下载上传完成'
