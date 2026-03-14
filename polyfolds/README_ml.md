# Polyfolds ML Scaffold

This folder now has an explicit ML scaffold separate from the older net-generation scripts.

## Purpose

The target is a hybrid raster + vector pipeline:

- raster input for image-based baselines and future CNN work
- vector structure for exact faces, edges, edge-pair colors, and repair targets

That is a better fit for Polyfolds than treating everything as pixels forever.

## New Pieces

- `polyfolds_ml/schema.py`
  - defines the manifest/sample schema
- `polyfolds_ml/manifest.py`
  - normalizes legacy `labels.jsonl` datasets into one manifest
- `polyfolds_ml/architecture.py`
  - stores explicit classifier and repair model specs
- `polyfolds_ml/training.py`
  - baseline classifier training scaffold from manifest data
- `build_polyfolds_manifest.py`
  - CLI to build a unified manifest
- `train_polyfolds_classifier.py`
  - CLI to train a first baseline classifier

## Current Limits

- legacy datasets do not yet provide edge-group colors everywhere
- repair targets are scaffolded, not fully generated
- the baseline classifier is a simple sklearn scaffold, not the final CNN
- the repair model is represented as a planned architecture, not implemented

## Immediate Next Step

Build a manifest from the current legacy datasets, inspect class balance, and then decide whether to:

1. train a quick baseline on existing raster data
2. extend the generators so every sample also emits a target repair SVG / vector spec
