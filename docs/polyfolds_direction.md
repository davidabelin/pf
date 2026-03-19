# Polyfolds Direction Tonight/Tomorrow

## Tonight's Cleanup
- Keep `pf_web` minimal and healthy.
- Make root-level tests and CLIs work from the repo root.
- Split runtime dependencies from offline geometry and ML dependencies.
- Freeze the new direction in docs so the legacy PNG corpora are not mistaken for canonical training data.

## Canonical Dataset
- Canonical data is vector-first.
- Each sample carries structured geometry JSON, a canonical SVG, and an optional preview PNG.
- `tetra`, `hexa`, and `octa` aim for exhaustive valid-topology coverage.
- `dodeca` and `icosa` start from deterministic sampled valid topologies and expand later without changing topology hashes.
- `valid`, `incomplete`, and `invalid` samples are grouped by one base topology hash so split leakage is prevented.

## Labels and Models
- The primary label space is `solid + state`.
- The first training track is a shared CNN classifier over on-the-fly rasterized views from the canonical vectors.
- The vector repair path stays phase 2 and does not add adversarial training until supervised vector targets are stable.

## Color Policy
- Canonical assets store semantic render roles now.
- Neutral rendering is the default until the explicit color-spec pass lands.
- Final semantic color mapping is intentionally deferred so geometry and interfaces can stabilize first.

## Legacy Status
- All legacy code and data moved to `polyfolds/legacy`
- `polyfolds/legacy/dataset_*` remains legacy and non-canonical.
- The sklearn baseline remains legacy scaffolding and no longer defines the main training path.
