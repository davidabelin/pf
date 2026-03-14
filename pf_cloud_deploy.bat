@echo off
setlocal
call "%~dp0pf_cloud_env.bat"
pushd "%~dp0" >nul 2>&1
if errorlevel 1 goto :fail

echo.
echo ==== Polyfolds App Engine Deploy ====
echo Project : %PROJECT_ID%
echo Service : %SERVICE_NAME%
echo Region  : %REGION%

echo.
echo [1/3] Previewing upload payload...
call gcloud meta list-files-for-upload > upload-list.txt
if errorlevel 1 goto :fail_popd
for /f %%i in ('find /c /v "" ^< upload-list.txt') do set UPLOAD_COUNT=%%i
echo Upload file count: %UPLOAD_COUNT%

echo.
echo [2/3] Deploying App Engine service...
call gcloud app deploy app.aix.yaml --project="%PROJECT_ID%" --quiet
if errorlevel 1 goto :fail_popd

echo.
echo [3/3] Listing deployed services...
call gcloud app services list --project="%PROJECT_ID%"
if errorlevel 1 goto :fail_popd

popd >nul
endlocal
exit /b 0

:fail_popd
popd >nul

:fail
echo.
echo [ERROR] Polyfolds deploy failed.
endlocal
exit /b 1
