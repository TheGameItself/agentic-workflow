@echo off
REM MCP Agentic Workflow Accelerator - USB Launcher for Windows
echo MCP Agentic Workflow Accelerator - USB Edition
echo ==============================================

cd /d "%~dp0"
cd platforms\windows

if exist "start_mcp.bat" (
    call start_mcp.bat
) else if exist "mcp_agentic_workflow.exe" (
    mcp_agentic_workflow.exe
) else (
    echo No Windows executable found
    pause
    exit /b 1
)
