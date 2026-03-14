from __future__ import annotations

from solid_common import DatasetDefaults, NetsDefaults
from solid_polyface import SolidSpec, main as polyface_main


SPEC = SolidSpec(
    key="tetra",
    polyhedra_id=1,
    faces_total=4,
    nets_max=2,
    defaults_dataset=DatasetDefaults(out_dir="dataset_tetra", line_width=2.0, workers=8, chunksize=20),
    defaults_nets=NetsDefaults(out_dir="nets_tetra", line_width=2.0, workers=8, chunksize=20),
    default_missing=1,
)


def main(argv: list[str] | None = None) -> None:
    polyface_main(SPEC, argv)


if __name__ == "__main__":
    main()

