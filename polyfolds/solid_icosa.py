from __future__ import annotations

from solid_common import DatasetDefaults, NetsDefaults
from solid_polyface import SolidSpec, main as polyface_main


SPEC = SolidSpec(
    key="icosa",
    polyhedra_id=5,
    faces_total=20,
    nets_max=43380,
    defaults_dataset=DatasetDefaults(out_dir="dataset_icosa", line_width=2.0, workers=8, chunksize=20),
    defaults_nets=NetsDefaults(out_dir="nets_icosa", line_width=2.0, workers=8, chunksize=20),
    default_missing=3,
)


def main(argv: list[str] | None = None) -> None:
    polyface_main(SPEC, argv)


if __name__ == "__main__":
    main()

