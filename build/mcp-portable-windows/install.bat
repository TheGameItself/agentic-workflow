@echo off
echo Installing MCP...
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
echo Installation complete!
pause
