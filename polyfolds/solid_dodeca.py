from __future__ import annotations

from solid_common import DatasetDefaults, NetsDefaults
from solid_polyface import SolidSpec, main as polyface_main


SPEC = SolidSpec(
    key="dodeca",
    polyhedra_id=4,
    faces_total=12,
    nets_max=43380,
    defaults_dataset=DatasetDefaults(out_dir="dataset_dodeca", line_width=2.0, workers=8, chunksize=20),
    defaults_nets=NetsDefaults(out_dir="nets_dodeca", line_width=2.0, workers=8, chunksize=20),
    default_missing=2,
)


def main(argv: list[str] | None = None) -> None:
    polyface_main(SPEC, argv)


if __name__ == "__main__":
    main()

