# Polyfolds Data Spec

## Canonical Sample Contract
- `sample_id`: stable sample identifier, currently `<solid>_<topology_hash>_<state>`
- `split`: `train`, `val`, or `test`
- `solid`: one of `tetra`, `hexa`, `octa`, `dodeca`, `icosa`
- `state`: one of `valid`, `incomplete`, `invalid`
- `joint_label`: `<solid>:<state>`
- `topology_hash`: stable family key used for split integrity
- `vector_json_path`: canonical geometry payload
- `canonical_svg_path`: canonical neutral SVG
- `render_profile_id`: currently `neutral_v1`
- `source_kind`: `canonical` or `legacy`
- `vector_faces` and `vector_edges`: normalized geometry carried into the unified manifest

## Canonical Root Layout
- `samples.jsonl`
- `dataset_manifest.json`
- `vector_json/<solid>/<sample_id>.json`
- `svg/<solid>/<sample_id>.svg`
- `preview/<solid>/<sample_id>.png` only when preview generation is requested

## Exemplar Construction Rules
- `valid`: a deterministic valid net from the local geometry code
- `incomplete`: derived from the valid net by removing leaf faces; missing faces are kept as completion targets
- `invalid`: derived from the valid net by overlap flip or subtree detachment
- All three states from one valid net reuse the same `topology_hash` and split

## Coverage Policy
- `tetra`, `hexa`, `octa`: exhaustive valid-topology coverage
- `dodeca`, `icosa`: deterministic sampled valid-topology coverage first, default cap `2048`
- Future expansion must preserve the already-issued topology hashes

## Render Profile `neutral_v1`
- background: `#ffffff`
- face fill: `#dbe0e7`
- face outline: `#182028`
- completion outline: `#707c8a`
- shared edge: `#6e7680`
- cut edge: `#182028`

## CLI
1. Build canonical data:
   - `python polyfolds\build_canonical_polyfolds_dataset.py --out-dir data\canonical_core`
2. Generate dataset QA plus exemplar contact sheet:
   - `python polyfolds\report_polyfolds_dataset.py data\canonical_core --output-json data\canonical_core\report.json --output-md data\canonical_core\report.md --contact-sheet data\canonical_core\exemplars.png`
3. Build unified manifest:
   - `python polyfolds\build_polyfolds_manifest.py --dataset data\canonical_core --output data\canonical_core\manifest.json --name polyfolds_canonical_core`

## External Search Conclusion
- A bounded search found vector sources, but not a usable labeled dataset of valid Polyfold nets with stable family IDs:
  - `https://polyhedronmodels.org/ks/svgfiles/`
  - `https://github.com/roni-polymod/models`
- Canonical data therefore remains locally generated.

## References
- See `docs/polyfolds_references.md` for unfolding and net-enumeration references that may inform later validation and enumeration refinements.
