@echo off
chcp 65001 >nul
title llama-modchip - DeepSeek-V4-Flash Fractal (Windows, GPU + CPU)
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0modchip_dsv4.ps1" %*
if errorlevel 1 (
    echo.
    echo [!] Server stopped with an error. Press any key to close...
    pause >nul
)
