@echo off
setlocal

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

rem Merges dataset_icosa_pale into dataset_icosa by renaming/moving PNGs and appending labels.jsonl.
"%PY%" "%~dp0merge_dataset_variant.py" --src "dataset_dodeca_pale" --dst "dataset_dodeca" --solid "dodeca" --variant "pale" --delete-src
if errorlevel 1 exit /b 1

echo Done.

