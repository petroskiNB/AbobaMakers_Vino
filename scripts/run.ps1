param(
    [switch]$Preview,
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [string]$BindAddress = '127.0.0.1'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Python environment missing. Run: python -m venv .venv; then .\.venv\Scripts\python.exe -m pip install -r requirements-web.txt (preview) or -r requirements.txt (recognition).'
}
if (-not $Preview) {
    $RequiredArtifacts = @(
        'artifacts/base_model/config.json',
        'artifacts/base_model/model.safetensors',
        'artifacts/base_model/preprocessor_config.json',
        'artifacts/base_model/source.json',
        'artifacts/experiment/index.pt',
        'artifacts/experiment/adapter_best.pt',
        'artifacts/experiment/manifest.json'
    )
    $MissingArtifacts = @($RequiredArtifacts | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $ProjectRoot $_))
    })
    if ($MissingArtifacts.Count -gt 0) {
        throw ("Recognition cannot start. Missing model files:`n" + ($MissingArtifacts -join "`n") +
            "`nCopy trained artifacts from your team's working project, or train with your data. For interface preview run: .\scripts\run.ps1 -Preview")
    }
}
$PreviousPreview = $env:WINE_PREVIEW
Push-Location $ProjectRoot
try {
    $env:WINE_PREVIEW = if ($Preview) { '1' } else { '0' }
    & $Python -m uvicorn wine_ml.api:app --host $BindAddress --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }
} finally {
    $env:WINE_PREVIEW = $PreviousPreview
    Pop-Location
}
