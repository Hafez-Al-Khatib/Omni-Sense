@echo off
REM flash_fleet.bat — Batch-flash multiple Omni-Sense ESP32 sensors (Windows)
REM =============================================================================
REM Usage:
REM   1. Connect sensor #1 via USB
REM   2. Run: flash_fleet.bat 01
REM   3. Disconnect #1, connect #2
REM   4. Run: flash_fleet.bat 02
REM   5. Repeat for each sensor
REM
REM If SENSOR_ID="AUTO" is set in config.h, simply run "pio run -t upload"
REM for every sensor and the ID derives from MAC automatically.

setlocal enabledelayedexpansion

set ID=%1
if "%ID%"=="" set ID=01
set SENSOR_ID=esp32-s3-%ID%

echo ==============================================
echo Flashing sensor: %SENSOR_ID%
echo ==============================================

cd /d "%~dp0\.."

pio run -t upload --build-flag "-DSENSOR_ID=\"%SENSOR_ID%\""

echo.
echo Sensor %SENSOR_ID% flashed successfully.
echo Unplug this sensor and plug in the next one.
pause
