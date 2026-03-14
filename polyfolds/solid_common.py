from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetDefaults:
    out_dir: str = "dataset"
    n_valid: int = 2000
    n_incomplete: int = 2000
    n_invalid: int = 2000
    seed: int = 2025
    img_size: int = 512
    palette: str = "pastel"
    line_width: float = 1.0
    workers: int = 8
    chunksize: int = 20


@dataclass(frozen=True)
class NetsDefaults:
    out_dir: str = "nets"
    count: int = 5
    seed: int = 2025
    sidelength: float = 1.0
    img_size: int = 512
    palette: str = "pastel"
    line_width: float = 1.0
    labels: bool = False
    max_tries: int = 5000
    no_png: bool = False
    workers: int = 8
    chunksize: int = 20


def add_fast_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fast", action="store_true", help="use all CPU cores where supported (forces workers=0)")


def add_dataset_args(p: argparse.ArgumentParser, *, defaults: DatasetDefaults) -> None:
    add_fast_flag(p)
    p.add_argument("--out-dir", default=defaults.out_dir, help=f"output directory (default: {defaults.out_dir})")
    p.add_argument("--n-valid", type=int, default=defaults.n_valid, help=f"number of valid samples (default: {defaults.n_valid})")
    p.add_argument(
        "--n-incomplete",
        type=int,
        default=defaults.n_incomplete,
        help=f"number of incomplete samples (default: {defaults.n_incomplete})",
    )
    p.add_argument("--n-invalid", type=int, default=defaults.n_invalid, help=f"number of invalid samples (default: {defaults.n_invalid})")
    p.add_argument("--seed", type=int, default=defaults.seed, help=f"random seed (default: {defaults.seed})")
    p.add_argument("--img-size", type=int, default=defaults.img_size, help=f"output PNG size (default: {defaults.img_size})")
    p.add_argument("--palette", default=defaults.palette, help=f"face color palette: pastel|random|gray (default: {defaults.palette})")
    p.add_argument("--line-width", type=float, default=defaults.line_width, help=f"outline line width in pixels (default: {defaults.line_width:g})")
    p.add_argument("--workers", type=int, default=defaults.workers, help=f"worker processes (0=all cores, default: {defaults.workers})")
    p.add_argument("--chunksize", type=int, default=defaults.chunksize, help=f"work chunk size per process (default: {defaults.chunksize})")
    p.add_argument("--plot", action="store_true", help="write preview image to <out-dir>/preview.png")
    p.add_argument("--test", action="store_true", help="run ~1/10th-sized dataset for quick iteration")


def add_nets_args(p: argparse.ArgumentParser, *, defaults: NetsDefaults) -> None:
    add_fast_flag(p)
    p.add_argument("--out-dir", default=defaults.out_dir, help=f"output directory (default: {defaults.out_dir})")
    p.add_argument("--count", type=int, default=defaults.count, help=f"number of nets to generate (default: {defaults.count})")
    p.add_argument("--seed", type=int, default=defaults.seed, help=f"base RNG seed (default: {defaults.seed})")
    p.add_argument("--sidelength", type=float, default=defaults.sidelength, help=f"edge length (default: {defaults.sidelength:g})")
    p.add_argument("--img-size", type=int, default=defaults.img_size, help=f"output PNG size (default: {defaults.img_size})")
    p.add_argument("--palette", default=defaults.palette, help=f"face color palette: pastel|random|gray (default: {defaults.palette})")
    p.add_argument("--line-width", type=float, default=defaults.line_width, help=f"outline line width in pixels (default: {defaults.line_width:g})")
    p.add_argument("--labels", action="store_true", default=defaults.labels, help="draw face index labels")
    p.add_argument("--max-tries", type=int, default=defaults.max_tries, help=f"max attempts per net before failing (default: {defaults.max_tries})")
    p.add_argument("--no-png", action="store_true", default=defaults.no_png, help="only write JSONL, skip PNG rendering")
    p.add_argument("--workers", type=int, default=defaults.workers, help=f"worker processes (0=all cores, default: {defaults.workers})")
    p.add_argument("--chunksize", type=int, default=defaults.chunksize, help=f"work chunk size per process (default: {defaults.chunksize})")


def apply_test_scale(n: int) -> int:
    return max(1, int(n) // 10)
