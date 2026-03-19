# Polyfolds Direction

## Workspace Split
- `pf_web/` stays lightweight and offline-first.
- `polyfolds/` is the real geometry, dataset, and training workspace.
- Legacy raster datasets and older generators live under `polyfolds/legacy/`.

## Data Decisions Frozen Today
- The canonical training source is local generation, not imported PNG corpora.
- Each canonical sample is vector-first and carries:
  - `vector_json/<solid>/<sample_id>.json`
  - `svg/<solid>/<sample_id>.svg`
  - `preview/<solid>/<sample_id>.png` only when requested
  - `samples.jsonl` plus `dataset_manifest.json`
- The normalized sample contract now requires `state`, `joint_label`, `topology_hash`, `vector_json_path`, `canonical_svg_path`, `render_profile_id`, and `source_kind`.
- `valid`, `incomplete`, and `invalid` samples are derived as one topology family and forced into the same split through `topology_hash`.
- `tetra`, `hexa`, and `octa` use exhaustive valid-topology coverage.
- `dodeca` and `icosa` start from deterministic sampled valid topologies with stable hashes so later expansion does not rename existing families.

## Input Visual Policy
- The current canonical render profile is `neutral_v1`.
- The exact palette is now fixed in code:
  - background: `#ffffff`
  - face fill: `#dbe0e7`
  - face outline: `#182028`
  - completion outline: `#707c8a`
  - shared edge: `#6e7680`
  - cut edge: `#182028`
- Render roles remain semantic so a later color redesign can change presentation without changing topology data.

## External Data Search Result
- A bounded search was completed for free vector sources with stable IDs and usable metadata.
- Two credible vector sources were found:
  - `https://polyhedronmodels.org/ks/svgfiles/`
  - `https://github.com/roni-polymod/models`
- Neither is a drop-in labeled dataset of valid Polyfold nets with stable topology-family IDs, so local generation remains the canonical path.

## Model Direction
- The first trainable model is one shared `solid:state` classifier with 15 labels.
- The baseline architecture remains:
  - input: `192x192` RGB raster rendered on demand from vector geometry
  - encoder: 4 conv stages with channels `32/64/128/256`
  - head: global average pooling then `256 -> 128 -> 15`
- Training now assumes:
  - topology-family split integrity with leak detection
  - balanced joint-label sampling by default
  - AdamW
  - affine-only augmentation
  - early stopping on validation macro-F1
- Vector repair remains phase 2 after the classifier and canonical dataset QA are stable.

## CLI Path
1. Build canonical data:
   - `python polyfolds\build_canonical_polyfolds_dataset.py --out-dir data\canonical_core`
2. Report and preview the dataset:
   - `python polyfolds\report_polyfolds_dataset.py data\canonical_core --output-json data\canonical_core\report.json --output-md data\canonical_core\report.md --contact-sheet data\canonical_core\exemplars.png`
3. Build a unified manifest:
   - `python polyfolds\build_polyfolds_manifest.py --dataset data\canonical_core --output data\canonical_core\manifest.json --name polyfolds_canonical_core`
4. Inspect the current model spec:
   - `python polyfolds\show_polyfolds_model_spec.py --which classifier`
5. Train the shared classifier:
   - `python polyfolds\train_polyfolds_classifier.py --manifest data\canonical_core\manifest.json --artifact models\polyfolds_cnn_classifier.pt`

## Supporting Notes
- See `docs/polyfolds_data_spec.md` for the canonical data contract and exemplar construction rules.
- See `docs/polyfolds_model_spec.md` for the shared classifier architecture and training defaults.
