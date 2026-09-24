$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = '.\.venv\Scripts\python.exe'
& $Python -X utf8 scripts\prepare_catalog.py --images data\foto --catalog data\strapi_output0709.csv
& $Python -X utf8 scripts\download_model.py
& $Python -X utf8 -m wine_ml.experiment extract
& $Python -X utf8 -m wine_ml.experiment train --epochs 15
& $Python -X utf8 scripts\calibrate.py
& $Python -X utf8 -m wine_ml.ocr
& $Python -X utf8 scripts\evaluate_hybrid.py
