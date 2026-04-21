$ErrorActionPreference = "Stop"

Set-Location "E:\Code Documentation Search"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Set-Content -Path ".\app_stdout.log" -Value ""
Set-Content -Path ".\app_stderr.log" -Value ""

& ".\.venv\Scripts\python.exe" -u "app.py" 2>&1 |
    Tee-Object -FilePath ".\app_stdout.log"
