@echo off
REM start.bat — Lanza backend Python y frontend Electron juntos (Windows)
REM Uso:
REM    start.bat                             -> sin sesión
REM    start.bat sessions\ejemplo_demo\config.json

SETLOCAL
SET PROJECT_ROOT=%~dp0
CD /D "%PROJECT_ROOT%"

IF EXIST ".venv\Scripts\activate.bat" (
  CALL .venv\Scripts\activate.bat
)

SET SESSION_CONFIG=%~1
IF "%SESSION_CONFIG%"=="" (
  START "GestDeck Backend" cmd /k "python -m core.main"
) ELSE (
  START "GestDeck Backend" cmd /k "python -m core.main --config ""%SESSION_CONFIG%"""
)

TIMEOUT /T 2 /NOBREAK >NUL

CD app
START "GestDeck Frontend" cmd /k "npm run dev"

ENDLOCAL
