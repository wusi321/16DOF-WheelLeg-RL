@echo off
title Full Course Preview - 14m x 6m
cd /d "D:\Project\nano\16dof-rl"
if errorlevel 1 goto badpath
set PYTHONPATH=D:\Project\nano\16dof-rl\src;D:\Project\nano\16dof-rl\mjlab\src
set WARP_CACHE_PATH=D:\Project\nano\16dof-rl\.cache\warp
set PYTHONIOENCODING=utf-8
"D:\Project\wheelleg-rl\.venv\Scripts\python.exe" "%~dp0view_full_course.py" %*
if errorlevel 1 goto failed
goto done
:badpath
echo [ERROR] Project folder not found: D:\Project\nano\16dof-rl
pause >nul
goto done
:failed
echo.
echo [ERROR] Python exited with an error. See the messages above.
pause >nul
:done
