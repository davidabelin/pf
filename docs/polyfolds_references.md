# Polyfolds References

Working bibliography for net enumeration, unfolding constraints, and polyhedra representation.

## Tracking Conventions
- `URL`: source landing page or stable reference link when known
- `PDF`: local copy status or direct PDF link when known
- `Status`: `not checked`, `citation only`, `located`, `downloaded`, or `reviewed`
- `Use`: why the reference matters to Polyfolds

## References

### Demaine, Demaine, Lubiw, O'Rourke (2002)
- Citation:
  - Demaine, Erik D.; Demaine, Martin L.; Lubiw, Anna; O'Rourke, Joseph (2002), "Enumerating foldings and unfoldings between polygons and polytopes", *Graphs and Combinatorics*, 18 (1): 93-104, arXiv:cs.CG/0107024, doi:10.1007/s003730200005, MR 1892436, S2CID 1489
- URL:
  - pending stable link capture
- PDF:
  - not downloaded
- Status:
  - citation only
- Use:
  - baseline reference for counting and characterizing foldings and unfoldings
  - useful for future refinement of canonical topology IDs and enumeration claims
- Notes:
  - likely one of the most directly relevant references for valid-net enumeration logic

### Miller and Pak (2008)
- Citation:
  - Miller, Ezra; Pak, Igor (2008), "Metric combinatorics of convex polyhedra: Cut loci and nonoverlapping unfoldings", *Discrete & Computational Geometry*, 39 (1-3): 339-388, doi:10.1007/s00454-008-9052-3, MR 2383765
- URL:
  - pending stable link capture
- PDF:
  - not downloaded
- Status:
  - citation only
- Use:
  - helpful for understanding nonoverlap constraints and the geometry behind legal unfoldings
  - likely relevant to future invalid-sample design and overlap validation rules

### O'Rourke (2011)
- Citation:
  - O'Rourke, Joseph (2011), *How to Fold It: The Mathematics of Linkages, Origami and Polyhedra*, Cambridge University Press, pp. 115-116, ISBN 9781139498548
- URL:
  - pending stable link capture
- PDF:
  - not downloaded
- Status:
  - citation only
- Use:
  - likely useful as a concise conceptual reference for nets and foldability examples
  - may help with explanatory notes and sanity checks rather than implementation details

### Demaine and O'Rourke (2007)
- Citation:
  - Demaine, Erik D.; O'Rourke, Joseph (2007), "Chapter 22. Edge Unfolding of Polyhedra", *Geometric Folding Algorithms: Linkages, Origami, Polyhedra*, Cambridge University Press, pp. 306-338
- URL:
  - pending stable link capture
- PDF:
  - not downloaded
- Status:
  - citation only
- Use:
  - broad implementation guide for edge-unfolding terminology, constraints, and algorithmic framing
  - likely useful when tightening the definition of valid/incomplete/invalid net families

### Malkevitch
- Citation:
  - Malkevitch, Joseph, "Nets: A Tool for Representing Polyhedra in Two Dimensions", *Feature Columns*, American Mathematical Society, retrieved 2014-05-14
- URL:
  - pending stable link capture
- PDF:
  - not downloaded
- Status:
  - citation only
- Use:
  - lightweight explanatory reference for representation choices and communication
  - useful for docs and user-facing descriptions rather than core algorithms

## Project Relevance Summary
- Enumeration:
  - Demaine et al. (2002)
- Nonoverlap and geometric legality:
  - Miller and Pak (2008)
  - Demaine and O'Rourke (2007)
- Exposition and representation:
  - O'Rourke (2011)
  - Malkevitch

## Next Actions
- Find stable URLs for each citation
- Download PDFs or source pages where legally available
- Add a `docs/references/` or `papers/` local storage convention before saving files into the repo
- Extract 3-5 implementation-relevant takeaways per reference after review
