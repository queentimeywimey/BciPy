<#
.SYNOPSIS
    Install Windows requirements for BciPy into a dedicated conda environment.

.DESCRIPTION
    Mirrors scripts/shell/linux_requirements.sh and scripts/shell/m2chip_install.sh
    for Windows. This script is intended to be run from the root of the BciPy
    repository, in a PowerShell terminal, with conda available on PATH.

    It will:
      - Create (or reuse) a conda environment with a Python version compatible
        with BciPy (>3.8,<3.11; Python 3.10 by default).
      - Check for the Microsoft Visual C++ Build Tools (required to build some
        BciPy dependencies from source) and offer to install them via winget.
      - Upgrade pip, setuptools and wheel inside that environment.
      - Install BciPy in editable mode, including dev dependencies, inside
        that environment.

.NOTES
    Requires conda (Anaconda/Miniconda) to already be installed and on PATH.
    Run from an elevated PowerShell prompt if winget needs to install the
    Visual C++ Build Tools.
#>

[CmdletBinding()]
param(
    # Name of the conda environment to create/use.
    [string]$EnvName = "bcipy",

    # Python version to install into the conda environment.
    [string]$PythonVersion = "3.10",

    # Skip installing the optional dev/test dependencies (`pip install -e .` only).
    [switch]$NoDev
)

$ErrorActionPreference = "Stop"

function Write-Step($message) {
    Write-Host "`n==> $message" -ForegroundColor Cyan
}

###### Verify conda is available ######
Write-Step "Checking for conda"

$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    throw "conda was not found on PATH. Install Anaconda/Miniconda from https://docs.conda.io/en/latest/miniconda.html before continuing."
}

###### Create (or reuse) the conda environment ######
Write-Step "Checking for conda environment '$EnvName'"

$envList = conda env list --json | ConvertFrom-Json
$envExists = $envList.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName }

if ($envExists) {
    Write-Host "Conda environment '$EnvName' already exists, reusing it."
} else {
    Write-Host "Creating conda environment '$EnvName' with Python $PythonVersion..."
    conda create -y -n $EnvName "python=$PythonVersion"
}

###### Microsoft Visual C++ Build Tools ######
Write-Step "Checking for Microsoft Visual C++ Build Tools"

$vcInstalled = $false
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $installed = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($installed) {
        $vcInstalled = $true
    }
}

if ($vcInstalled) {
    Write-Host "Visual C++ Build Tools already installed."
} else {
    Write-Warning "Microsoft Visual C++ Build Tools were not detected."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host "Installing Visual C++ Build Tools via winget (this may take a while)..."
        winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-package-agreements --accept-source-agreements
    } else {
        Write-Warning "winget not found. Please install the Visual C++ Build Tools manually: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    }
}

###### Upgrade packaging tools inside the conda environment ######
Write-Step "Upgrading pip, setuptools and wheel in '$EnvName'"
conda run -n $EnvName python -m pip install --upgrade pip setuptools wheel

###### Install BciPy inside the conda environment ######
Write-Step "Installing BciPy in editable mode in '$EnvName'"
if ($NoDev) {
    conda run -n $EnvName python -m pip install -e .
} else {
    conda run -n $EnvName python -m pip install -e ".[dev]"
}

Write-Step "Done. Run 'conda activate $EnvName' then 'python bcipy/gui/BCInterface.py' to launch the GUI, or 'bcipy --help' for CLI usage."
