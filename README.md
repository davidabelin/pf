# Polyfolds
Standalone Polyfolds sister app plus the offline geometry and ML lab that feeds it.

## Layout
- `pf_web/` is the deployed Flask shell.
- `polyfolds/` is the offline workspace for geometry, canonical dataset generation, manifests, and training.
- `data/` is the local output area for generated canonical datasets.
- `models/` is the local output area for trained artifacts.
- `tests/` holds repo smoke tests.

## Current Direction
- The runtime app stays lightweight and offline-first for now.
- The canonical training source is a vector-first dataset generated under `data/canonical_core/`.
- The existing `polyfolds/legacy/dataset_*` PNG corpora are legacy reference assets only.
- The first model track is one shared `solid + state` classifier, followed later by vector repair.

## Local Workflow
1. Install runtime-only deps with `pip install -r requirements.txt` if you only need the Flask shell.
2. Install the offline workspace with `pip install -r requirements-dev.txt` for geometry, manifests, tests, and training.
3. Generate canonical vector-first data with `python polyfolds\\build_canonical_polyfolds_dataset.py --out-dir data\\canonical_core`.
4. Report the dataset and generate exemplar previews with `python polyfolds\\report_polyfolds_dataset.py data\\canonical_core --output-json data\\canonical_core\\report.json --output-md data\\canonical_core\\report.md --contact-sheet data\\canonical_core\\exemplars.png`.
5. Build a unified manifest with `python polyfolds\\build_polyfolds_manifest.py --dataset data\\canonical_core --output data\\canonical_core\\manifest.json --name polyfolds_canonical_core`.
6. Inspect the current classifier spec with `python polyfolds\\show_polyfolds_model_spec.py --which classifier`.
7. Train the shared classifier with `python polyfolds\\train_polyfolds_classifier.py --manifest data\\canonical_core\\manifest.json --artifact models\\polyfolds_cnn_classifier.pt`.

## Legacy Data
The checked-out folders under `polyfolds/legacy/dataset_*` remain useful for inspection and migration, but they are not the canonical source for new training runs. The manifest builder will still ingest them as legacy datasets when needed.

## Cloud Flow
1. Run `pf_cloud_setup.bat` once per project.
2. Run `pf_cloud_deploy.bat` to deploy the `polyfolds` service.
3. Run `pf_cloud_status.bat` to verify service versions and bucket state.
