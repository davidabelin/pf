import os, json, random, math
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Optional: better overlap detection
try:
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

Vec2 = Tuple[float, float]

# -----------------------------
# Regular polygon templates (2D)
# -----------------------------

def regular_polygon(n: int, radius: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Returns Nx2 vertices CCW."""
    pts = []
    for k in range(n):
        a = phase + 2*math.pi*k/n
        pts.append((radius*math.cos(a), radius*math.sin(a)))
    return np.array(pts, dtype=float)

def equilateral_triangle(side: float = 1.0) -> np.ndarray:
    # triangle with side length 1, CCW
    # vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    h = math.sqrt(3)/2
    pts = np.array([[0,0],[1,0],[0.5,h]], dtype=float)
    return pts * side

# -----------------------------
# Polyhedron definition by faces (vertex indices)
# We'll use ONLY face adjacency + shared-edge mapping.
# -----------------------------

@dataclass(frozen=True)
class Polyhedron:
    name: str
    faces: List[Tuple[int, ...]]  # each face is tuple of vertex indices (ordered CCW)
    face_sides: int               # p (triangle=3)
    # computed:
    adj: Dict[int, List[int]] = None
    shared_edge: Dict[Tuple[int,int], Tuple[int,int]] = None
    # shared_edge[(fa,fb)] = (edge_index_in_fa, edge_index_in_fb)

def build_adjacency(poly: Polyhedron) -> Polyhedron:
    faces = poly.faces
    p = poly.face_sides

    # Map undirected edge -> (face_id, edge_index, oriented endpoints)
    edge_map = {}  # key: frozenset({u,v}) -> list of (f, ei, u, v)
    for fi, f in enumerate(faces):
        for ei in range(p):
            u = f[ei]
            v = f[(ei+1) % p]
            key = tuple(sorted((u,v)))
            edge_map.setdefault(key, []).append((fi, ei, u, v))

    shared_edge = {}
    adj = defaultdict(list)

    for key, uses in edge_map.items():
        if len(uses) == 2:
            (f1,e1, u1,v1), (f2,e2, u2,v2) = uses
            adj[f1].append(f2)
            adj[f2].append(f1)
            shared_edge[(f1,f2)] = (e1,e2)
            shared_edge[(f2,f1)] = (e2,e1)

    return Polyhedron(poly.name, poly.faces, poly.face_sides, dict(adj), shared_edge)

def tetrahedron() -> Polyhedron:
    # 4 vertices, 4 triangular faces
    # Use consistent orientation (not crucial for unfolding; only for shared-edge indices)
    faces = [
        (0,1,2),
        (0,3,1),
        (1,3,2),
        (2,3,0),
    ]
    return build_adjacency(Polyhedron("tetra", faces, 3))

def octahedron() -> Polyhedron:
    # 6 vertices, 8 triangular faces
    # One common combinatorial labeling:
    # vertices 0=+Z, 1=-Z, 2=+X,3=-X,4=+Y,5=-Y (conceptually)
    faces = [
        (0,2,4),
        (0,4,3),
        (0,3,5),
        (0,5,2),
        (1,4,2),
        (1,3,4),
        (1,5,3),
        (1,2,5),
    ]
    return build_adjacency(Polyhedron("octa", faces, 3))

# -----------------------------
# Random spanning tree on a graph (Wilson's algorithm)
# -----------------------------

def random_spanning_tree(nodes: List[int], neighbors: Dict[int, List[int]]) -> List[Tuple[int,int]]:
    """
    Returns list of undirected edges (u,v) forming a random spanning tree.
    Wilson's algorithm (loop-erased random walk).
    """
    root = random.choice(nodes)
    in_tree = {root}
    parent = {}

    while len(in_tree) < len(nodes):
        start = random.choice([n for n in nodes if n not in in_tree])
        path = [start]
        seen_index = {start: 0}
        cur = start

        while cur not in in_tree:
            nxt = random.choice(neighbors[cur])
            if nxt in seen_index:
                # erase loop
                loop_start = seen_index[nxt]
                path = path[:loop_start+1]
                seen_index = {v:i for i,v in enumerate(path)}
            else:
                path.append(nxt)
                seen_index[nxt] = len(path)-1
            cur = nxt

        # add path to tree
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            parent[u] = v
            in_tree.add(u)

    edges = []
    for u, v in parent.items():
        a, b = sorted((u,v))
        edges.append((a,b))
    return edges

# -----------------------------
# Planar placement of faces from a spanning tree
# (Each child face is placed by a rigid transform that glues its shared edge
#  to the parent's shared edge, with the face on the opposite side.)
# -----------------------------

def side_of_line(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    # signed area *2 of triangle (a,b,p)
    return float((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))

def similarity_transform(p0, p1, q0, q1, reflect=False):
    """
    Return function T(x) applying similarity mapping:
    p0->q0, p1->q1, with optional reflection across local x-axis.
    """
    p0 = np.array(p0, float); p1=np.array(p1,float)
    q0 = np.array(q0, float); q1=np.array(q1,float)
    vp = p1 - p0
    vq = q1 - q0
    lp = np.linalg.norm(vp)
    lq = np.linalg.norm(vq)
    if lp < 1e-12 or lq < 1e-12:
        raise ValueError("degenerate segment")
    s = lq / lp
    # local basis for p
    ex = vp / lp
    ey = np.array([-ex[1], ex[0]])
    # local basis for q
    Ex = vq / lq
    Ey = np.array([-Ex[1], Ex[0]])
    # matrix mapping local coords in p to local coords in q
    # [Ex Ey] * [x; y] with optional flip of y
    M = np.stack([Ex, Ey], axis=1)
    def T(x):
        x = np.array(x, float)
        u = x - p0
        # coords in (ex,ey)
        cx = np.dot(u, ex)
        cy = np.dot(u, ey)
        if reflect:
            cy = -cy
        vec = M @ np.array([cx*s, cy*s])
        return q0 + vec
    return T

def unfold_tree(poly: Polyhedron, tree_edges: List[Tuple[int,int]], side_len: float=1.0) -> Optional[Dict[int, np.ndarray]]:
    """
    Returns dict face_id -> 2D polygon vertices (p x 2) or None if inconsistent.
    """
    p = poly.face_sides
    # local template for a face in its own coordinates
    if p == 3:
        local = equilateral_triangle(side_len)
    else:
        # placeholder for later (pentagon etc.)
        local = regular_polygon(p, radius=1.0)

    # build tree adjacency
    tadj = defaultdict(list)
    for u,v in tree_edges:
        tadj[u].append(v)
        tadj[v].append(u)

    # root face
    faces = list(range(len(poly.faces)))
    root = random.choice(faces)
    placed = {root: local.copy()}

    # BFS
    q = deque([root])
    parent = {root: None}

    while q:
        f = q.popleft()
        Pf = placed[f]

        for g in tadj[f]:
            if g == parent.get(f):  # skip parent link
                continue
            if g in placed:
                continue
            parent[g] = f

            # shared edge indices (in face f and face g)
            ef, eg = poly.shared_edge[(f,g)]
            # endpoints of shared edge in f's placed polygon
            a = Pf[ef]
            b = Pf[(ef+1) % p]

            # We will place g by mapping its local shared edge to segment (b->a) (reversed)
            # so that the two faces lie on opposite sides of the edge.
            Pg0 = local.copy()

            # local edge endpoints in g
            p0 = Pg0[eg]
            p1 = Pg0[(eg+1) % p]

            # Two candidate transforms: reflect False/True
            # Both map p0->b and p1->a. Choose the one that puts g on opposite side from f.
            T0 = similarity_transform(p0, p1, b, a, reflect=False)
            T1 = similarity_transform(p0, p1, b, a, reflect=True)

            cand0 = np.array([T0(x) for x in Pg0])
            cand1 = np.array([T1(x) for x in Pg0])

            # Determine which side f lies on relative to edge a-b:
            # pick f's third vertex (one not on edge ef)
            f_third = Pf[(ef+2) % p]  # for triangles
            sf = side_of_line(a, b, f_third)

            # For each candidate, pick its third vertex and compute side
            g_third0 = cand0[(eg+2) % p]
            sg0 = side_of_line(a, b, g_third0)

            g_third1 = cand1[(eg+2) % p]
            sg1 = side_of_line(a, b, g_third1)

            # Want opposite signs (strictly)
            chosen = None
            if sf * sg0 < 0:
                chosen = cand0
            elif sf * sg1 < 0:
                chosen = cand1
            else:
                # Degenerate or numerical weirdness; try whichever is "more opposite"
                chosen = cand0 if abs(sf*sg0) < abs(sf*sg1) else cand1

            placed[g] = chosen
            q.append(g)

    return placed

# -----------------------------
# Overlap test
# -----------------------------

def polygons_overlap(polys: List[np.ndarray], eps: float=1e-9) -> bool:
    """
    Returns True if any two polygons overlap with positive area.
    Touching along edges/vertices is allowed.
    """
    if HAS_SHAPELY:
        sp = [SPoly(p) for p in polys]
        for i in range(len(sp)):
            for j in range(i+1, len(sp)):
                inter = sp[i].intersection(sp[j])
                if inter.area > 1e-8:
                    return True
        return False

    # Fallback: raster-ish check using matplotlib path sampling
    # (cruder but works for triangles reasonably; you can replace with SAT later)
    from matplotlib.path import Path
    paths = [Path(p) for p in polys]
    # sample points inside each polygon (triangle): barycentric grid
    for i in range(len(polys)):
        pi = polys[i]
        # sample 15 points in i
        pts = []
        if len(pi) == 3:
            A,B,C = pi
            for a in np.linspace(0.1,0.8,5):
                for b in np.linspace(0.1,0.8,5):
                    if a+b < 0.9:
                        pts.append(A*a + B*b + C*(1-a-b))
        else:
            # generic: sample bounding box
            minxy = pi.min(axis=0); maxxy = pi.max(axis=0)
            for _ in range(80):
                pts.append(np.random.uniform(minxy, maxxy))
        pts = np.array(pts)
        for j in range(len(polys)):
            if i==j: continue
            # if any interior sample of i lies inside j, treat as overlap
            if np.any(paths[j].contains_points(pts)):
                return True
    return False

# -----------------------------
# Valid net test for tetra/octa
# -----------------------------

def is_valid_net_for_poly(poly: Polyhedron, tree_edges: List[Tuple[int,int]]) -> bool:
    placed = unfold_tree(poly, tree_edges, side_len=1.0)
    if placed is None: 
        return False
    polys = [placed[i] for i in range(len(poly.faces))]
    return not polygons_overlap(polys)

# -----------------------------
# Rendering 512x512 with style randomization
# -----------------------------

def render_faces(out_path: str, faces2d: Dict[int, np.ndarray], img_size=512):
    polys = list(faces2d.values())
    all_pts = np.vstack(polys)
    minxy = all_pts.min(axis=0)
    maxxy = all_pts.max(axis=0)
    span = maxxy - minxy

    # padding + fit to canvas
    pad = random.uniform(40, 90)
    target = img_size - 2*pad
    s = min(target/span[0], target/span[1]) if min(span) > 1e-9 else 1.0

    # random rotation
    ang = math.radians(random.choice([0, 60, 120, 180, 240, 300]) + random.uniform(-7,7))
    R = np.array([[math.cos(ang), -math.sin(ang)],
                  [math.sin(ang),  math.cos(ang)]], float)

    # transform polys
    tpolys = []
    for P in polys:
        Q = (P - minxy) * s
        Q = (Q - Q.mean(axis=0)) @ R.T + Q.mean(axis=0)
        tpolys.append(Q)
    allq = np.vstack(tpolys)
    minq = allq.min(axis=0)
    maxq = allq.max(axis=0)

    ox = random.uniform(pad*0.5, img_size - (maxq[0]-minq[0]) - pad*0.5)
    oy = random.uniform(pad*0.5, img_size - (maxq[1]-minq[1]) - pad*0.5)

    edge_w = random.uniform(0.5, 2.0)   # random.uniform(2.0, 8.0)
    alpha = random.uniform(0.33, 0.67) # random.uniform(0.65,1.0)

    fig = plt.figure(figsize=(img_size/100, img_size/100), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor((1,1,1))

    for Q in tpolys:
        QQ = Q - minq + np.array([ox, oy])
        fc = (random.uniform(0.05,0.95), random.uniform(0.05,0.95), random.uniform(0.05,0.95), alpha)
        ax.add_patch(Polygon(QQ, closed=True, facecolor=fc, edgecolor=(0,0,0,1), linewidth=edge_w))

    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

# -----------------------------
# Dataset generation for tetra+octa
# -----------------------------
# Originally: max_tries=2000
def sample_valid_tree(poly: Polyhedron, max_tries=500) -> Tuple[List[Tuple[int,int]], Dict[int,np.ndarray]]:
    nodes = list(range(len(poly.faces)))
    for _ in range(max_tries):
        tree = random_spanning_tree(nodes, poly.adj)
        placed = unfold_tree(poly, tree, side_len=1.0)
        if placed and (not polygons_overlap([placed[i] for i in nodes])):
            return tree, placed
    raise RuntimeError(f"Could not find valid net for {poly.name} in {max_tries} tries")

def sample_invalid_tree(poly: Polyhedron, max_tries=500) -> Tuple[List[Tuple[int,int]], Dict[int,np.ndarray]]:
    nodes = list(range(len(poly.faces)))
    for _ in range(max_tries):
        tree = random_spanning_tree(nodes, poly.adj)
        placed = unfold_tree(poly, tree, side_len=1.0)
        if placed and polygons_overlap([placed[i] for i in nodes]):
            return tree, placed
    raise RuntimeError(f"Could not find invalid(net-overlap) for {poly.name} in {max_tries} tries")

def make_incomplete_from_valid(poly: Polyhedron, valid_tree: List[Tuple[int,int]], placed: Dict[int,np.ndarray], k_remove_faces: int = 1):
    """
    Incomplete = remove k faces from placed net. Provide completion face ids as target.
    """
    face_ids = list(placed.keys())
    # keep it connected-ish: remove leaves preferred
    deg = defaultdict(int)
    for u,v in valid_tree:
        deg[u]+=1; deg[v]+=1
    leaves = [f for f in face_ids if deg[f] == 1]
    removal = random.sample(leaves, k=min(k_remove_faces, len(leaves))) if leaves else random.sample(face_ids, k=k_remove_faces)
    remain = {f: P for f,P in placed.items() if f not in removal}
    completion = removal
    return remain, completion

def write_jsonl(fh, rec: dict):
    fh.write(json.dumps(rec) + "\n")

def split_of(i, total):
    r = i / max(1, total)
    return "train" if r < 0.8 else ("val" if r < 0.9 else "test")

def make_dataset(out_dir="dataset_tri", n_each=2000, seed=2025):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    labels_path = os.path.join(out_dir, "labels.jsonl")
    fh = open(labels_path, "w", encoding="utf-8")

    polys = [tetrahedron(), octahedron()]

    idx = 0
    for poly in polys:
        # VALID
        for i in range(n_each):
            tree, placed = sample_valid_tree(poly)
            fname = f"valid_{poly.name}_{i:07d}.png"
            fpath = os.path.join(img_dir, fname)
            render_faces(fpath, placed, img_size=512)
            write_jsonl(fh, {
                "file": os.path.join("images", fname),
                "split": split_of(i, n_each),
                "class": "valid",
                "solid": poly.name,
                "tree_edges": tree,
                "completion_faces": [],
            })
            idx += 1

        # INCOMPLETE
        for i in range(n_each):
            tree, placed = sample_valid_tree(poly)
            k = random.choice([1,1,1,2])
            remain, completion = make_incomplete_from_valid(poly, tree, placed, k_remove_faces=k)
            fname = f"incomplete_{poly.name}_{i:07d}.png"
            fpath = os.path.join(img_dir, fname)
            render_faces(fpath, remain, img_size=512)
            write_jsonl(fh, {
                "file": os.path.join("images", fname),
                "split": split_of(i, n_each),
                "class": "incomplete",
                "solid": poly.name,
                "tree_edges": tree,
                "completion_faces": completion,   # supervision target
            })
            idx += 1

        # INVALID (overlap-producing unfoldings)
        for i in range(n_each):
            tree, placed = sample_invalid_tree(poly)
            fname = f"invalid_{poly.name}_{i:07d}.png"
            fpath = os.path.join(img_dir, fname)
            render_faces(fpath, placed, img_size=512)
            write_jsonl(fh, {
                "file": os.path.join("images", fname),
                "split": split_of(i, n_each),
                "class": "invalid",
                "solid": poly.name,
                "tree_edges": tree,
                "completion_faces": [],
            })
            idx += 1

    fh.close()
    print("Wrote:", labels_path)
    print("Images in:", img_dir)
    print("Shapely overlap:", HAS_SHAPELY)

if __name__ == "__main__":
    make_dataset(out_dir="dataset_tri", n_each=500, seed=507)  # original: n_each=1500
