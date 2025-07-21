@echo off
REM MCP Agentic Workflow Accelerator - Unwrap Script (Windows)
REM This script extracts and sets up the MCP system from the archive

echo 📦 MCP Agentic Workflow Accelerator - Unwrap Script
echo ==================================================

REM Check if archive exists
if not exist "mcp_system.zip" (
    echo ❌ Archive not found: mcp_system.zip
    echo    Make sure you're running this script from the deployment package directory.
    pause
    exit /b 1
)

REM Get target directory name
if not "%1"=="" (
    set TARGET_DIR=%1
) else (
    set TARGET_DIR=mcp_workflow_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
    set TARGET_DIR=%TARGET_DIR: =0%
)

echo 📁 Target directory: %TARGET_DIR%

REM Create target directory
mkdir "%TARGET_DIR%"

REM Extract archive
echo 📦 Extracting archive...
if "mcp_system.zip" == *.zip (
    powershell -command "Expand-Archive -Path 'mcp_system.zip' -DestinationPath '%TARGET_DIR%' -Force"
) else if "mcp_system.zip" == *.tar.gz (
    tar -xzf "mcp_system.zip" -C "%TARGET_DIR%"
) else (
    echo ❌ Unsupported archive format: mcp_system.zip
    pause
    exit /b 1
)

echo ✅ Archive extracted successfully

REM Navigate to target directory
cd "%TARGET_DIR%"

REM Run setup
echo 🔧 Running setup...
if exist "setup.bat" (
    setup.bat
) else if exist "setup.sh" (
    echo ⚠️  Linux setup script found. Please run './setup.sh' manually on Linux/Mac.
) else (
    echo ⚠️  No setup script found. Installing dependencies manually...
    pip install -r requirements.txt
    python test_system.py
)

echo.
echo 🎉 MCP Agentic Workflow Accelerator is ready!
echo 📁 Location: %TARGET_DIR%
echo.
echo 📋 Next steps:
echo   cd %TARGET_DIR%
echo   python mcp_cli.py --help
echo   python test_system.py
echo.
echo 📖 Read README.md for detailed usage instructions
pause
