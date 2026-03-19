@echo off
setlocal
call "%~dp0pf_cloud_env.bat"
pushd "%~dp0" >nul 2>&1
if errorlevel 1 goto :fail

echo.
echo ==== Polyfolds Cloud Setup ====
echo Project : %PROJECT_ID%
echo Service : %SERVICE_NAME%
echo Region  : %REGION%
echo Bucket  : %BUCKET_NAME%
echo SA      : %SA_EMAIL%

echo.
echo [1/5] Setting active gcloud project...
call gcloud config set project "%PROJECT_ID%"
if errorlevel 1 goto :fail_popd

echo.
echo [2/5] Enabling required APIs...
call gcloud services enable appengine.googleapis.com storage.googleapis.com --project="%PROJECT_ID%"
if errorlevel 1 goto :fail_popd

echo.
echo [3/5] Ensuring App Engine app exists...
call gcloud app describe --project="%PROJECT_ID%" >nul 2>nul
if errorlevel 1 (
  echo ^> gcloud app create --project="%PROJECT_ID%" --region="%REGION%"
  call gcloud app create --project="%PROJECT_ID%" --region="%REGION%"
  if errorlevel 1 goto :fail_popd
) else (
  echo [OK] App Engine app already exists.
)

echo.
echo [4/5] Ensuring storage bucket exists...
call gcloud storage buckets describe "gs://%BUCKET_NAME%" --project="%PROJECT_ID%" >nul 2>nul
if errorlevel 1 (
  echo ^> gcloud storage buckets create "gs://%BUCKET_NAME%" --project="%PROJECT_ID%" --location="%REGION%" --uniform-bucket-level-access
  call gcloud storage buckets create "gs://%BUCKET_NAME%" --project="%PROJECT_ID%" --location="%REGION%" --uniform-bucket-level-access
  if errorlevel 1 goto :fail_popd
) else (
  echo [OK] Bucket already exists.
)
call gcloud storage buckets update "gs://%BUCKET_NAME%" --project="%PROJECT_ID%" --public-access-prevention
if errorlevel 1 goto :fail_popd

echo.
echo [5/5] Granting bucket access to the runtime service account...
call gcloud storage buckets add-iam-policy-binding "gs://%BUCKET_NAME%" --member="serviceAccount:%SA_EMAIL%" --role="roles/storage.objectAdmin" --quiet
if errorlevel 1 goto :fail_popd

echo.
echo [OK] Polyfolds cloud setup finished.
popd >nul
endlocal
exit /b 0

:fail_popd
popd >nul

:fail
echo.
echo [ERROR] Polyfolds cloud setup failed.
endlocal
exit /b 1
