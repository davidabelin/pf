from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

Cell = Tuple[int, int]
Vec = Tuple[int, int, int]

DIRS: dict[str, tuple[int, int]] = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}
ALL_NORMALS: Set[Vec] = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}


def neighbors4(c: Cell) -> List[Cell]:
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def normalize(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    minx, miny = min(xs), min(ys)
    norm = sorted((x - minx, y - miny) for x, y in cells)
    return tuple(norm)


def transform_cell(c: Cell, k: int, reflect: bool) -> Cell:
    x, y = c
    if reflect:
        x = -x
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y)


def canonical_free(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    cells = list(cells)
    variants = []
    for reflect in [False, True]:
        for k in range(4):
            v = [transform_cell(c, k, reflect) for c in cells]
            variants.append(normalize(v))
    return min(variants)


def is_connected(cells: Set[Cell]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        c = q.popleft()
        for n in neighbors4(c):
            if n in cells and n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen) == len(cells)


def enumerate_free_polyominoes(n: int) -> List[Tuple[Cell, ...]]:
    polys: Set[Tuple[Cell, ...]] = {canonical_free([(0, 0)])}
    for _size in range(2, n + 1):
        new_set: Set[Tuple[Cell, ...]] = set()
        for poly in polys:
            cells = set(poly)
            boundary = set()
            for c in cells:
                for nb in neighbors4(c):
                    if nb not in cells:
                        boundary.add(nb)
            for b in boundary:
                grown = set(cells)
                grown.add(b)
                can = canonical_free(grown)
                new_set.add(can)
        polys = new_set
    return sorted(polys)


def vneg(a: Vec) -> Vec:
    return (-a[0], -a[1], -a[2])


def v_eq(a: Vec, b: Vec) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@dataclass(frozen=True)
class Frame:
    u: Vec
    v: Vec
    n: Vec


def step_frame(fr: Frame, dir2: str) -> Frame:
    u, v, n = fr.u, fr.v, fr.n
    if dir2 == "E":
        return Frame(u=vneg(n), v=v, n=u)
    if dir2 == "W":
        return Frame(u=n, v=v, n=vneg(u))
    if dir2 == "N":
        return Frame(u=u, v=vneg(n), n=v)
    if dir2 == "S":
        return Frame(u=u, v=n, n=vneg(v))
    raise ValueError(dir2)


def foldable_cube_net_discrete(poly: Iterable[Cell]) -> bool:
    cells = set(poly)
    if len(cells) != 6 or not is_connected(cells):
        return False

    adj: Dict[Cell, List[Tuple[Cell, str]]] = defaultdict(list)
    for x, y in cells:
        for d, (dx, dy) in DIRS.items():
            nb = (x + dx, y + dy)
            if nb in cells:
                adj[(x, y)].append((nb, d))

    root = next(iter(cells))
    root_fr = Frame(u=(1, 0, 0), v=(0, 1, 0), n=(0, 0, 1))
    frame: Dict[Cell, Frame] = {root: root_fr}
    used_norms: Set[Vec] = {root_fr.n}

    q = deque([root])
    while q:
        c = q.popleft()
        fr = frame[c]
        for nb, d in adj[c]:
            fr_nb = step_frame(fr, d)
            if nb in frame:
                if not (
                    v_eq(frame[nb].n, fr_nb.n)
                    and v_eq(frame[nb].u, fr_nb.u)
                    and v_eq(frame[nb].v, fr_nb.v)
                ):
                    return False
                continue
            if fr_nb.n in used_norms:
                return False
            frame[nb] = fr_nb
            used_norms.add(fr_nb.n)
            q.append(nb)

    return used_norms == ALL_NORMALS


__all__ = [
    "ALL_NORMALS",
    "Cell",
    "DIRS",
    "enumerate_free_polyominoes",
    "foldable_cube_net_discrete",
    "is_connected",
]

