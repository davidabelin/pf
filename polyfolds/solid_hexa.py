from __future__ import annotations

import argparse
from typing import List

from solid_common import DatasetDefaults, NetsDefaults, add_dataset_args, add_nets_args
from solid_polyface import SolidSpec, run_nets


_NETS_SPEC = SolidSpec(
    key="hexa",
    polyhedra_id=3,  # cube
    faces_total=6,
    nets_max=11,
    defaults_dataset=DatasetDefaults(out_dir="dataset_hexa"),
    defaults_nets=NetsDefaults(out_dir="nets_hexa"),
    default_missing=1,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hexahedron (cube) tools")
    sub = p.add_subparsers(dest="cmd")

    nets = sub.add_parser("nets", help="Generate unfolded 2D hexahedron nets (random non-overlapping)")
    add_nets_args(nets, defaults=_NETS_SPEC.defaults_nets)

    ds = sub.add_parser("dataset", help="Generate hexahedron (cube net) square datasets (wraps datagen_squares.py)")
    add_dataset_args(ds, defaults=DatasetDefaults(out_dir="dataset_hexa", line_width=2.0, workers=8, chunksize=20))

    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd == "nets":
        run_nets(_NETS_SPEC, args)
        return

    if args.cmd == "dataset":
        import datagen_squares

        workers = 0 if bool(args.fast) else int(args.workers)
        argv2 = [
            "--out-dir",
            str(args.out_dir),
            "--n-valid",
            str(int(args.n_valid)),
            "--n-incomplete",
            str(int(args.n_incomplete)),
            "--n-invalid",
            str(int(args.n_invalid)),
            "--seed",
            str(int(args.seed)),
            "--img-size",
            str(int(args.img_size)),
            "--palette",
            str(args.palette),
            "--line-width",
            str(float(args.line_width)),
            "--workers",
            str(int(workers)),
            "--chunksize",
            str(int(args.chunksize)),
            "--solid-name",
            "hexa",
        ]
        if args.test:
            argv2.append("--test")
        if args.plot:
            argv2.append("--plot")

        datagen_squares.main(argv2)
        return

    build_parser().print_help()


if __name__ == "__main__":
    main()
