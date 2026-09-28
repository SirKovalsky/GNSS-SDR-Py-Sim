@echo off
rem Build native/gnss_sim_tx.exe (MSVC + UHD import library).
rem
rem Requires: MSVC (Visual Studio 2019 BuildTools or newer) and UHD installed
rem (UHD_PKG_PATH, or the default C:\Program Files\UHD).
setlocal enabledelayedexpansion
pushd "%~dp0"

if not defined UHD_PKG_PATH set "UHD_PKG_PATH=C:\Program Files\UHD"
if not exist "%UHD_PKG_PATH%\include\uhd" (
    echo [build] error: UHD not found at "%UHD_PKG_PATH%" ^(set UHD_PKG_PATH^)
    popd & exit /b 1
)

set "VCVARS="
for %%V in (
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if not defined VCVARS if exist %%V set "VCVARS=%%~V"
)
if not defined VCVARS (
    echo [build] error: vcvars64.bat not found ^(install MSVC Build Tools^)
    popd & exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 (
    echo [build] error: failed to initialise MSVC environment
    popd & exit /b 1
)

echo [build] UHD: %UHD_PKG_PATH%
echo [build] compiling tx_player.cpp ...
cl /nologo /std:c++14 /EHsc /MD /O2 /D_CRT_SECURE_NO_WARNINGS ^
    /I"%UHD_PKG_PATH%\include" ^
    tx_player.cpp ^
    /Fe:gnss_sim_tx.exe ^
    /link /LIBPATH:"%UHD_PKG_PATH%\lib" uhd.lib
if errorlevel 1 (
    echo [build] FAILED
    popd & exit /b 1
)

del /q *.obj >nul 2>&1
echo [build] OK: %~dp0gnss_sim_tx.exe
echo [build] run it with: set PATH=%UHD_PKG_PATH%\bin;%%PATH%%  ^&  gnss_sim_tx.exe --file ... --rate ... --freq ...
popd
endlocal
exit /b 0
