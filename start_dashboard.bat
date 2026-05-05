@echo off
cd /d "%~dp0"
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
".runtime\python312\python.exe" start_dashboard.py
