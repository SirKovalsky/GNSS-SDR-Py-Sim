#Requires -Version 5.1
<#
.SYNOPSIS
    Standalone setup for GNSS Sim on Windows.

.DESCRIPTION
    Creates a project-local virtual environment (.venv), installs the Python
    requirements, and optionally installs the UHD Python bindings (when the
    native UHD libraries are present) and CuPy (NVIDIA CUDA).  Idempotent:
    re-running reuses an existing .venv and only installs what is missing.

    No external project is referenced; everything lives inside this repository.

.PARAMETER Python
    Command used to create the venv.  Default: "py -3.13".

.PARAMETER Cuda
    Force-install "cupy-cuda12x<14" for CUDA 12.x GPU synthesis.

.EXAMPLE
    .\scripts\setup_windows.ps1
.EXAMPLE
    .\scripts\setup_windows.ps1 -Cuda
#>
[CmdletBinding()]
param(
    [string]$Python = "py -3.13",
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== GNSS Sim setup ==" -ForegroundColor Cyan
Write-Host "Project: $Root"

# --- 1. Project-local virtual environment ---------------------------------
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    Write-Host "Reusing existing .venv"
} else {
    Write-Host "Creating .venv ($Python -m venv .venv)"
    $parts = $Python -split '\s+'
    $exe = $parts[0]
    $rest = @()
    if ($parts.Length -gt 1) { $rest = $parts[1..($parts.Length - 1)] }
    & $exe @rest -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) {
        throw "Failed to create .venv. Install Python 3.10-3.13 or pass -Python <exe>."
    }
}

# --- 2. Python dependencies ----------------------------------------------
Write-Host "Upgrading pip"
& $VenvPy -m pip install --upgrade pip
Write-Host "Installing requirements.txt"
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }

# --- 3. UHD (only when the native libraries are installed) ----------------
function Find-UhdDll {
    $candidates = @()
    if ($env:UHD_PKG_PATH) {
        $candidates += (Join-Path $env:UHD_PKG_PATH "bin\uhd.dll")
        $candidates += (Join-Path $env:UHD_PKG_PATH "uhd.dll")
    }
    $candidates += "C:\Program Files\UHD\bin\uhd.dll"
    $candidates += "C:\Program Files (x86)\UHD\bin\uhd.dll"
    foreach ($p in $candidates) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    foreach ($dir in ($env:Path -split ';')) {
        if ($dir -and (Test-Path (Join-Path $dir "uhd.dll"))) {
            return (Join-Path $dir "uhd.dll")
        }
    }
    return $null
}

$uhdDll = Find-UhdDll
if ($uhdDll) {
    Write-Host "UHD found ($uhdDll) - installing uhd==4.10.0.0"
    & $VenvPy -m pip install "uhd==4.10.0.0"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Could not install the UHD Python bindings; file-only mode still works."
    }
} else {
    Write-Host "UHD not found - skipping. For USRP B210 TX:"
    Write-Host "  1) install UHD for Windows (https://files.ettus.com/binaries/uhd/latest_release/)"
    Write-Host "  2) re-run this script, or: .venv\Scripts\python.exe -m pip install uhd==4.10.0.0"
}

# --- 4. CUDA / CuPy (optional) --------------------------------------------
$nvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($Cuda) {
    Write-Host "Installing CuPy for CUDA 12.x (cupy-cuda12x<14)"
    & $VenvPy -m pip install "cupy-cuda12x<14"
} elseif ($nvidia) {
    Write-Host "NVIDIA GPU detected - for GPU synthesis re-run with -Cuda, or:"
    Write-Host "  .venv\Scripts\python.exe -m pip install 'cupy-cuda12x<14'"
}

# --- 5. Done --------------------------------------------------------------
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run the GUI : .\.venv\Scripts\python.exe run.py"
Write-Host "Run the CLI : .\.venv\Scripts\python.exe -m gnss_sim --help"
Write-Host "Run tests   : .\.venv\Scripts\python.exe -m pytest tests -q"
