@echo off
setlocal
call "%~dp0pf_cloud_env.bat"

echo.
echo ==== Polyfolds Cloud Status ====
echo Project: %PROJECT_ID%
echo Service: %SERVICE_NAME%
echo Bucket : %BUCKET_NAME%

echo.
call gcloud app services list --project="%PROJECT_ID%"
if errorlevel 1 goto :fail

echo.
call gcloud app versions list --project="%PROJECT_ID%"
if errorlevel 1 goto :fail

echo.
call gcloud storage buckets list --project="%PROJECT_ID%" --format="table(name,location,storageClass)"
if errorlevel 1 goto :fail

endlocal
exit /b 0

:fail
echo.
echo [ERROR] Polyfolds status check failed.
endlocal
exit /b 1
