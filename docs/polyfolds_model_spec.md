# Polyfolds Model Spec

## Classifier Objective
- One shared classifier over the 15-label space `solid:state`
- Labels:
  - `tetra|hexa|octa|dodeca|icosa`
  - crossed with `valid|incomplete|invalid`

## Input
- Rasterized on demand from canonical vector geometry
- Default size: `192x192`
- Color mode: RGB
- Render profile: `neutral_v1`

## Architecture
- Conv stage 1: `3 -> 32`, kernel `3`, batch norm, ReLU, max-pool
- Conv stage 2: `32 -> 64`, kernel `3`, batch norm, ReLU, max-pool
- Conv stage 3: `64 -> 128`, kernel `3`, batch norm, ReLU, max-pool
- Conv stage 4: `128 -> 256`, kernel `3`, batch norm, ReLU, max-pool
- Head:
  - adaptive average pooling
  - flatten
  - linear `256 -> 128`
  - ReLU
  - dropout `0.15`
  - linear `128 -> 15`

## Training Defaults
- optimizer: AdamW
- learning rate: `3e-4`
- weight decay: `1e-4`
- batch size: `16`
- max epochs: `12`
- early stopping patience: `4`
- sampler: balanced weighted-random sampling over `joint_label`
- loss:
  - default balanced path: plain cross-entropy with balanced sampling
  - unbalanced fallback: inverse-frequency weighted cross-entropy
- split policy: topology-family split with leak detection

## Augmentation
- rotation: up to `12` degrees
- scale: `0.92` to `1.08`
- translation: up to `0.08` of frame width/height

## CLI
1. Print the current classifier spec:
   - `python polyfolds\show_polyfolds_model_spec.py --which classifier`
2. Train the classifier:
   - `python polyfolds\train_polyfolds_classifier.py --manifest data\canonical_core\manifest.json --artifact models\polyfolds_cnn_classifier.pt`

## Current Checkpoint
- A non-smoke day-one run was completed on `168` canonical samples built from:
  - `tetra`: 2 valid families
  - `hexa`: 11 valid families
  - `octa`: 11 valid families
  - `dodeca`: 16 valid families
  - `icosa`: 16 valid families
- Command shape:
  - `python polyfolds\train_polyfolds_classifier.py --manifest data\canonical_core_day1\manifest.json --artifact models\polyfolds_cnn_classifier_day1.pt --epochs 4 --patience 2 --batch-size 8`
- Result:
  - validation macro-F1: `0.1778`
  - test accuracy: `0.2000`
  - test macro-F1: `0.0576`
- Interpretation:
  - the pipeline is working end to end
  - the dataset is still too small and skewed for meaningful classifier quality
  - the next real lever is more coverage for `dodeca` and `icosa`, not more architectural complexity yet

## Phase 2
- Repair stays separate from the classifier
- The current repair contract points to target SVG plus completion-face metadata
- Adversarial training is deferred until supervised vector repair works

## References
- See `docs/polyfolds_references.md` for geometry and unfolding references that may inform later repair-model targets and validity constraints.
