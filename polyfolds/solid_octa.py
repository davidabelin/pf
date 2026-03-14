from __future__ import annotations

from solid_common import DatasetDefaults, NetsDefaults
from solid_polyface import SolidSpec, main as polyface_main


SPEC = SolidSpec(
    key="octa",
    polyhedra_id=2,
    faces_total=8,
    nets_max=11,
    defaults_dataset=DatasetDefaults(out_dir="dataset_octa", line_width=2.0, workers=8, chunksize=20),
    defaults_nets=NetsDefaults(out_dir="nets_octa", line_width=2.0, workers=8, chunksize=20),
    default_missing=2,
)


def main(argv: list[str] | None = None) -> None:
    polyface_main(SPEC, argv)


if __name__ == "__main__":
    main()

