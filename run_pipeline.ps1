#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$SoloDashboard,
    [switch]$NoInstalar,
    [switch]$NoDashboard,
    [switch]$OmitirShap,
    [int]$Hilos = -1,
    [ValidateRange(0, 10000)][int]$Bootstrap = 500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "El comando fallo ($LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

$Python = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    Invoke-Checked -Executable "python" -Arguments @("-m", "venv", ".venv")
}
Invoke-Checked -Executable $Python -Arguments @(
    "-c", "import sys; sys.exit('Se requiere Python 3.11 o posterior') if sys.version_info < (3, 11) else None"
)

if (-not $NoInstalar) {
    Invoke-Checked -Executable $Python -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
}

if (-not $SoloDashboard) {
    $Raw = Join-Path $PSScriptRoot "data/adolescentes_ensanut2024_w.csv"
    if (-not (Test-Path $Raw)) { throw "No se encontro la base original: $Raw" }
    Invoke-Checked -Executable $Python -Arguments @("-m", "unittest", "discover", "-s", "tests", "-v")
    $PipelineArguments = @("-m", "suicide_predictors", "run", "--jobs", "$Hilos", "--bootstrap", "$Bootstrap")
    if ($OmitirShap) { $PipelineArguments += "--skip-shap" }
    Invoke-Checked -Executable $Python -Arguments $PipelineArguments
    Write-Host "Pipeline completado. output/ contiene los unicos resultados vigentes, organizados por etapa."
}

if (-not $NoDashboard) {
    $Launch = $SoloDashboard -or ((Read-Host "Abrir dashboard? [s/N]") -match "^[sS]$")
    if ($Launch) {
        Invoke-Checked -Executable $Python -Arguments @("-m", "streamlit", "run", "dashboard/app.py")
    }
}
