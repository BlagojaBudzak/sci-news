@echo off
REM ============================================================
REM Sci News — scheduled pipeline run
REM Point Windows Task Scheduler's "Program" at this file.
REM Edit PROJECT_DIR below to match your actual PyCharm project path.
REM ============================================================

set PROJECT_DIR=C:\Users\YOURNAME\PycharmProjects\sci-news-aggregator
cd /d "%PROJECT_DIR%"

call venv\Scripts\activate.bat

REM Make sure Ollama's server is up. Harmless if it's already running
REM (e.g. via the Ollama desktop app) — this just no-ops or errors quietly.
start "" /min ollama serve
timeout /t 5 /nobreak >nul

python main.py --categories chemistry physics seismology >> logs\pipeline.log 2>&1

REM Publish the refreshed digest to GitHub Pages.
git add site\digests data\digests
git commit -m "Auto digest update %date% %time%"
git push origin main

call venv\Scripts\deactivate.bat
