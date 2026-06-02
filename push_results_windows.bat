@echo off
set SCRIPT_NAME=%1
set SRC_IMAGES=%2
set REPO=C:\Dirk\Neethling\iems\master\Lb-Python

copy /Y "C:\Dirk\Neethling\iems\master\PythonLBCourse\%SCRIPT_NAME%.py" "%REPO%\scripts\freesurface\"

xcopy /E /I /Y "%SRC_IMAGES%" "%REPO%\results\freesurface\%SCRIPT_NAME%\"

cd /D "%REPO%"
git add -A
git commit -m "Auto-push %SCRIPT_NAME% %DATE% %TIME%"
git push