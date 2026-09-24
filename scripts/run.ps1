$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
& .\.venv\Scripts\python.exe -m uvicorn wine_ml.api:app --host 127.0.0.1 --port 8080
