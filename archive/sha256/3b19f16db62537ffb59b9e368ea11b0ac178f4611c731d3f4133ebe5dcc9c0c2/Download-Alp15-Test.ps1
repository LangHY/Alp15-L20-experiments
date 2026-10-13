param(
    [switch]$Download,
    [long]$ConfirmBytes = 0,
    [string]$Output = 'E:\alp15-small-test-949',
    [string]$Venv = 'E:\alp15-download-venv'
)

# Compatible with Windows PowerShell 5.1. Default mode never installs/downloads.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$TotalBytes = [long]54382445102
$HubVersion = '1.31.0'
$ScriptFile = Join-Path $PSScriptRoot 'download_alp15_windows.py'

function Assert-SafeDirectory([string]$Value) {
    if ($Value -notmatch '^[A-Za-z]:\\' -or $Value -match '(^|[\\/])\.\.([\\/]|$)') {
        throw 'Use an absolute local drive directory without .. components.'
    }
    $Full = [IO.Path]::GetFullPath($Value).TrimEnd('\')
    if ($Full -eq [IO.Path]::GetPathRoot($Full).TrimEnd('\') -or
        $Full -eq [Environment]::GetFolderPath('UserProfile').TrimEnd('\')) {
        throw 'Drive root and user home are forbidden targets.'
    }
    $Current = [IO.DirectoryInfo]::new($Full)
    while ($null -ne $Current) {
        if (Test-Path -LiteralPath $Current.FullName) {
            $Item = Get-Item -LiteralPath $Current.FullName -Force
            if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or -not $Item.PSIsContainer) {
                throw 'Directory or ancestor is a reparse point / non-directory; preserved.'
            }
        }
        $Current = $Current.Parent
    }
    return $Full
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Command failed with exit code ' + $LASTEXITCODE + '; files preserved.') }
}

try {
    if (-not (Test-Path -LiteralPath $ScriptFile -PathType Leaf)) { throw 'Keep all package files beside this PowerShell script.' }
    $PythonCommand = Get-Command py -ErrorAction SilentlyContinue
    $PythonArgs = @('-3')
    if ($null -eq $PythonCommand) {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        $PythonArgs = @()
    }
    if ($null -eq $PythonCommand) { throw 'Install 64-bit Python 3.10 or newer (3.12 recommended), then run again.' }
    $PythonExe = $PythonCommand.Source
    Invoke-Checked $PythonExe ($PythonArgs + @('-c', "import sys,struct; sys.exit(0 if sys.version_info >= (3,10) and struct.calcsize('P')==8 else 1)"))
    # Validate the frozen bundle before any venv creation or package install.
    Invoke-Checked $PythonExe ($PythonArgs + @($ScriptFile, '--output', $Output))
    if (-not $Download) { exit 0 }
    if ($ConfirmBytes -ne $TotalBytes) { throw 'Explicit download requires -ConfirmBytes 54382445102.' }
    $Output = Assert-SafeDirectory $Output
    $Venv = Assert-SafeDirectory $Venv
    if ($Output -eq $Venv -or $Venv.StartsWith($Output + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $Output.StartsWith($Venv + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Data and venv must be separate directories.'
    }
    $VenvPython = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Venv)) {
        Invoke-Checked $PythonExe ($PythonArgs + @('-m', 'venv', $Venv))
    } elseif (-not (Test-Path -LiteralPath (Join-Path $Venv 'pyvenv.cfg') -PathType Leaf)) {
        throw 'Existing venv path is not a Python venv; preserved.'
    }
    # Inspect every existing venv entry before executing it or installing into it.
    $PendingDirectories = [Collections.Generic.Stack[string]]::new()
    $PendingDirectories.Push($Venv)
    while ($PendingDirectories.Count -gt 0) {
        $CurrentDirectory = $PendingDirectories.Pop()
        foreach ($Item in @(Get-ChildItem -LiteralPath $CurrentDirectory -Force)) {
            if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Venv contains a reparse point; preserved.'
            }
            if ($Item.PSIsContainer) { $PendingDirectories.Push($Item.FullName) }
        }
    }
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) { throw 'Venv Python is missing; preserved.' }
    # Validate the actual interpreter, not only the existence of venv files.
    $VenvCheck = "import os,pathlib,struct,sys; norm=lambda p:os.path.normcase(str(pathlib.Path(p).resolve())); cfg=pathlib.Path(sys.argv[1])/'pyvenv.cfg'; settings=dict((k.strip().lower(),v.strip().lower()) for line in cfg.read_text(encoding='utf-8').splitlines() for k,sep,v in [line.partition('=')] if sep); ok=norm(sys.prefix)==norm(sys.argv[1]) and norm(sys.base_prefix)!=norm(sys.prefix) and sys.version_info >= (3,10) and struct.calcsize('P')==8 and settings.get('include-system-site-packages')=='false'; sys.exit(0 if ok else 1)"
    & $VenvPython -c $VenvCheck $Venv *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'Venv identity/version/isolation check failed; directory preserved, no package installation.'
    }
    $env:HF_HUB_DISABLE_XET = '1'
    $env:HF_ENDPOINT = 'https://huggingface.co'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    Invoke-Checked $VenvPython @('-m', 'pip', 'install', '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple', ('huggingface_hub==' + $HubVersion))
    $AuthCheck = "from huggingface_hub import HfApi,get_token; import sys; t=get_token(); sys.exit(2) if not t else None; HfApi(endpoint='https://huggingface.co').whoami(token=t)"
    & $VenvPython -c $AuthCheck *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'No valid/reachable Hugging Face authentication. Login manually in this venv:'
        Write-Host ('& "' + (Join-Path $Venv 'Scripts\hf.exe') + '" auth login')
        Write-Host 'Use the interactive prompt. Accept the NVIDIA dataset terms on Hugging Face if requested.'
        throw 'Authentication check failed; ZIP download not started.'
    }
    Invoke-Checked $VenvPython @($ScriptFile, '--download', '--confirm-bytes', [string]$TotalBytes, '--output', $Output)
    exit 0
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
