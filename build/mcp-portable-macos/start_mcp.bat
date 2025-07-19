@echo off
REM MCP Agentic Workflow Accelerator - Windows Launcher Script
REM This script starts the MCP server with proper environment setup

echo 🚀 Starting MCP Agentic Workflow Accelerator...
echo ==============================================

REM Get the directory where this script is located
cd /d "%~dp0"

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found. Using system Python.
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo 📦 Checking dependencies...
    python -m pip install -r requirements.txt --quiet
)

REM Check if the main CLI script exists
if exist "mcp_cli.py" (
    echo 🎯 Starting MCP CLI...
    python mcp_cli.py %*
) else if exist "src\mcp\cli.py" (
    echo 🎯 Starting MCP CLI from src...
    python -m src.mcp.cli %*
) else (
    echo ❌ MCP CLI script not found
    echo Available files:
    dir *.py 2>nul || echo No .py files in current directory
    dir src\mcp\*.py 2>nul || echo No .py files in src\mcp\
    pause
    exit /b 1
) 