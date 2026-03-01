@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  build_installer.bat — Build DetectorInventario_Setup.exe
::  Run from the repo root:  scripts\build_installer.bat
:: ============================================================

set "REPO_ROOT=%~dp0.."
cd /d "%REPO_ROOT%"

echo.
echo =====================================================
echo   Detector de Inventario — Build Pipeline
echo =====================================================
echo.

:: ── Step 0: Verify Python ────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Activate your virtual environment first.
    pause & exit /b 1
)

:: ── Step 1: Install / upgrade PyInstaller ────────────────────────────────────
echo [1/4] Installing PyInstaller...
pip install --upgrade pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller.
    pause & exit /b 1
)
echo       Done.

:: ── Step 2: Generate icon ────────────────────────────────────────────────────
echo [2/4] Generating icon...
python scripts\generate_icon.py
if errorlevel 1 (
    echo [ERROR] Icon generation failed.
    pause & exit /b 1
)

:: ── Step 3: PyInstaller ──────────────────────────────────────────────────────
echo [3/4] Running PyInstaller (this may take several minutes)...
pyinstaller detector_inventario.spec --clean --noconfirm
if errorlevel 1 (
    echo [ERROR] PyInstaller failed. Check the output above.
    pause & exit /b 1
)
echo       Bundle created: dist\DetectorInventario\

:: ── Step 4: Inno Setup ───────────────────────────────────────────────────────
echo [4/4] Compiling Inno Setup installer...

:: Try common Inno Setup install locations
set "ISCC="
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
) else if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
    set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)

if "!ISCC!"=="" (
    echo [ERROR] Inno Setup 6 not found.
    echo         Download from https://jrsoftware.org/isdl.php and install it.
    pause & exit /b 1
)

"!ISCC!" installer\setup.iss
if errorlevel 1 (
    echo [ERROR] Inno Setup compilation failed.
    pause & exit /b 1
)

:: ── Done ─────────────────────────────────────────────────────────────────────
echo.
echo =====================================================
echo   BUILD COMPLETE
echo =====================================================
echo.
echo   Installer: %REPO_ROOT%\installer\output\DetectorInventario_Setup.exe
echo.
pause
endlocal
