# Polyfolds ML Workspace

This folder now has a canonical vector-first workflow separate from the older raster-heavy generators.

## Purpose

The target is a hybrid raster + vector pipeline:

- raster views rendered on demand for the shared CNN classifier
- vector structure for exact faces, edges, semantic render roles, and repair targets

That is a better fit for Polyfolds than treating everything as pixels forever.

## New Pieces

- `build_canonical_polyfolds_dataset.py`
  - CLI to build the canonical vector-first dataset
- `canonical_dataset.py`
  - deterministic valid/incomplete/invalid family generation around stable topology hashes
- `vector_render.py`
  - neutral SVG/PNG rendering shared by canonical generation and training
- `polyfolds_ml/schema.py`
  - defines the manifest/sample schema
- `polyfolds_ml/manifest.py`
  - normalizes both legacy `labels.jsonl` datasets and canonical `samples.jsonl` datasets into one manifest
- `polyfolds_ml/architecture.py`
  - stores explicit classifier and repair model specs
- `polyfolds_ml/training.py`
  - trains the shared PyTorch CNN classifier from manifest data
- `build_polyfolds_manifest.py`
  - CLI to build a unified manifest
- `train_polyfolds_classifier.py`
  - CLI to train the shared solid-plus-state classifier

## Current Limits

- final semantic color mapping is still deferred
- repair targets are present in v1 form but the vector repair model is not implemented yet
- hard solids (`dodeca`, `icosa`) start from deterministic sampled valid topologies before any full expansion
- the repair model is represented as a planned architecture, not implemented

## Immediate Next Step

1. Build a canonical dataset:

   `python build_canonical_polyfolds_dataset.py --out-dir ..\\data\\canonical_core`

2. Build a unified manifest from that canonical root:

   `python build_polyfolds_manifest.py --dataset ..\\data\\canonical_core --output ..\\data\\canonical_core\\manifest.json`

3. Train the shared classifier:

   `python train_polyfolds_classifier.py --manifest ..\\data\\canonical_core\\manifest.json --artifact ..\\models\\polyfolds_cnn_classifier.pt`

The checked-out `dataset_*` folders remain legacy reference assets only.
