# Polyfolds Rolling TO DO List

## General Housekeeping
- [x] Repo-root `pytest` and CLI entrypoints work from `pf/`
- [x] Runtime requirements split from offline geometry and ML requirements
- [x] Legacy raster assets and older generators moved under `polyfolds/legacy/`
- [x] Top-level docs describe the two-lane workspace accurately
- [ ] Add a short changelog or release note flow for dataset/model revisions

## Data Generation
- [x] Canonical dataset builder CLI
- [x] Dataset report CLI with JSON, Markdown, and exemplar contact-sheet output
- [x] Canonical input contract documented
- [x] Canonical render colors fixed explicitly for `neutral_v1`
- [x] Valid, incomplete, and invalid samples derived from shared topology families
- [x] Manifest/schema carry vector JSON paths, canonical SVG paths, topology hashes, and repair targets
- [x] Bounded web search for reusable vector sources completed
- [ ] Found a drop-in external labeled dataset of valid Polyfold nets
- [x] Default path remains local canonical generation because the external sources are not suitable as-is
- [ ] Human approval pass on generated exemplar sheet
- [ ] Expand `dodeca` and `icosa` beyond the initial sampled valid-topology cap when compute time is available

## Training Data Balance
- [x] Small solids keep exhaustive valid coverage
- [x] Large solids start sampled with stable topology hashes
- [x] Balanced joint-label sampling added to training
- [x] Topology-family split leakage check added to training
- [ ] Decide whether to keep balanced sampling only or add a second curriculum stage for real-frequency evaluation

## Models
- [x] Shared CNN classifier input spec documented
- [x] Shared CNN classifier architecture documented in code and docs
- [x] Training hyperparameters and augmentation budget formalized in code
- [x] CLI to print current model and training specs
- [x] Run an initial non-smoke canonical training job and log results in docs
- [ ] Run a larger canonical training job after increasing `dodeca` and `icosa` coverage
- [ ] Add evaluation export with per-label precision, recall, and confusion figures

## Repair / Generator Phase
- [x] Repair target contract frozen at `target_svg_path` plus completion-face metadata
- [ ] Implement the vector repair model
- [ ] Decide whether repair predicts final SVG directly or a structured edit program first
- [ ] Revisit adversarial training only after supervised vector repair works
