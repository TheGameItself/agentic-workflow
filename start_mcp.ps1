# MCP Agentic Workflow Accelerator - PowerShell Launcher Script
# This script starts the MCP server with proper environment setup

Write-Host "🚀 Starting MCP Agentic Workflow Accelerator..." -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if virtual environment exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  No virtual environment found. Using system Python." -ForegroundColor Yellow
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or later" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt --quiet
}

# Check if the main CLI script exists
if (Test-Path "mcp_cli.py") {
    Write-Host "🎯 Starting MCP CLI..." -ForegroundColor Green
    python mcp_cli.py $args
} elseif (Test-Path "src\mcp\cli.py") {
    Write-Host "🎯 Starting MCP CLI from src..." -ForegroundColor Green
    python -m src.mcp.cli $args
} else {
    Write-Host "❌ MCP CLI script not found" -ForegroundColor Red
    Write-Host "Available files:" -ForegroundColor Yellow
    Get-ChildItem *.py -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name)" }
    Get-ChildItem src\mcp\*.py -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name)" }
    Read-Host "Press Enter to exit"
    exit 1
} 