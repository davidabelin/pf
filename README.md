# pf
Standalone Polyfolds sister app for AIX.

## Layout
- `pf_web/` user-facing Flask app
- `polyfolds/` offline data generation and training workspace
- `tests/` app smoke tests
- `app.aix.yaml` App Engine service manifest
- `pf_cloud_setup.bat` first-time bucket and IAM setup
- `pf_cloud_deploy.bat` App Engine deploy helper
- `pf_cloud_status.bat` service and bucket status helper

## Intended split
The deployed app serves trained-model interactions.
The sibling `polyfolds/` folder remains the one-time development workspace for dataset generation, manifests, training, and evaluation.

## Cloud flow
1. Run `pf_cloud_setup.bat` once per project.
2. Run `pf_cloud_deploy.bat` to deploy the `polyfolds` service.
3. Run `pf_cloud_status.bat` to verify service versions and bucket state.
