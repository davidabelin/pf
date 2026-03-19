# Polyfolds Review Checklist

Reviewer: ______________________________

Date: ______________________________

Scope reviewed: ______________________________

Use one section per decision. Mark one of the two boxes, then write comments or requested changes.

---

## 1. Repo Hygiene

Review focus:
Root `pytest` passes and the import-path bootstrap does not feel too brittle.

Files to inspect:
- `conftest.py`
- `polyfolds/bootstrap_paths.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 2. Legacy Split

Review focus:
`polyfolds/legacy/` is the right home for old raster datasets and helper modules.

Files to inspect:
- `docs/polyfolds_direction.md`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 3. Canonical Sample Contract

Review focus:
Required fields and directory layout are correct for the long-term canonical dataset.

Files to inspect:
- `docs/polyfolds_data_spec.md`
- `polyfolds/polyfolds_ml/schema.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 4. Vector-First Decision

Review focus:
`vector JSON + canonical SVG + optional preview PNG` is the right canonical format.

Files to inspect:
- `polyfolds/canonical_dataset.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 5. Topology-Family Handling

Review focus:
`valid`, `incomplete`, and `invalid` sharing one `topology_hash` is the right split-leakage policy.

Files to inspect:
- `polyfolds/canonical_dataset.py`
- `polyfolds/polyfolds_ml/training.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 6. Incomplete Exemplar Rule

Review focus:
Removing leaf faces is the right construction rule for incomplete samples.

Files to inspect:
- `polyfolds/solid_polyface.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 7. Invalid Exemplar Rule

Review focus:
Overlap-flip and subtree-detach are the right invalid constructions.

Files to inspect:
- `polyfolds/canonical_dataset.py`
- `polyfolds/solid_polyface.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 8. Color and Render Policy

Review focus:
The neutral palette and semantic role separation are correct.

Files to inspect:
- `polyfolds/vector_render.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 9. Dataset Report CLI

Review focus:
The JSON report, Markdown report, and exemplar contact sheet are useful and sufficient.

Files to inspect:
- `polyfolds/dataset_report.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 10. Manifest Normalization

Review focus:
Canonical and legacy manifests are normalized the way you want.

Files to inspect:
- `polyfolds/polyfolds_ml/manifest.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 11. Model Objective

Review focus:
The shared 15-label `solid:state` classifier is the right first target.

Files to inspect:
- `docs/polyfolds_model_spec.md`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 12. Classifier Architecture

Review focus:
The CNN shape should be kept as-is or revised before larger training.

Files to inspect:
- `polyfolds/polyfolds_ml/architecture.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 13. Training Regimen

Review focus:
Balanced sampling, AdamW, augmentation budget, patience, and split policy are appropriate.

Files to inspect:
- `polyfolds/polyfolds_ml/training.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 14. CLI Usability

Review focus:
The end-to-end command flow feels right: build dataset, report dataset, build manifest, print model spec, train classifier.

Files to inspect:
- `README.md`
- `polyfolds/README_ml.md`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 15. Docs Alignment

Review focus:
The direction doc and rolling TODO match what you want next.

Files to inspect:
- `docs/polyfolds_direction.md`
- `docs/polyfolds_TO_DO.md`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 16. Test Changes

Review focus:
The updated web smoke test matches the intended footer contract.

Files to inspect:
- `tests/test_app.py`

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## 17. Overall Verdict

Review focus:
The dataset decisions are resolved enough to start scaling generation and training.

Decision:
- [ ] Accept
- [ ] Change

Comments / requested changes:

____________________________________________________________

____________________________________________________________

____________________________________________________________

---

## Final Summary

Highest-priority changes to make next:

1. __________________________________________________________

2. __________________________________________________________

3. __________________________________________________________

Sign-off:

____________________________________________________________
