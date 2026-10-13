# Backward-compatible entry point. The previous SCP script is preserved as *_scp_legacy.ps1.
[CmdletBinding()]
param([switch]$ProbeOnly, [switch]$SelfTest)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'upload_training_chunk0000_resumable.ps1') -ProbeOnly:$ProbeOnly -SelfTest:$SelfTest
