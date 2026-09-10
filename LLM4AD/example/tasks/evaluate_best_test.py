"""Evaluate the final record in sample_best.json on task test sets.

Examples:
  python example/tasks/evaluate_best_test.py --task tsp --logs logs/mcts_recipe_tsp \
      --data dataset/tsplib --output tsp_test.csv
  python example/tasks/evaluate_best_test.py --task cvrp --logs logs/mcts_recipe_cvrp \
      --data dataset/cvrplib/data/cvrp_test_lt200.npz --output cvrp_test.csv
  python example/tasks/evaluate_best_test.py --task vrptw --logs logs/mcts_recipe_vrptw \
      --data dataset/vrptw_test/solomon --output vrptw_test.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import traceback
import zipfile
from pathlib import Path

import numpy as np


def best_program(log_dir: Path) -> tuple[dict, str]:
    path = log_dir if log_dir.name == "sample_best.json" else log_dir / "sample_best.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]
    if not records:
        raise ValueError(f"No records in {path}")
    record = records[-1]
    code = record.get("program") or record.get("code")
    if not code:
        raise ValueError(f"Final record in {path} has no program/code")
    return record, code


def load_function(code: str):
    ns = {"np": np, "numpy": np, "math": math}
    exec(code, ns)
    for name in ("select_next_node", "select_next_job"):
        if callable(ns.get(name)):
            return ns[name]
    raise ValueError("Best program defines neither select_next_node nor select_next_job")


def normalize_tsp_datasets(payload):
    """Normalize historical TSPLIB pickle wrappers and field names."""
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise ValueError(f"TSP pickle root must be a dictionary, got {type(payload).__name__}")
    for wrapper in ("tsplib_dict", "tsp_dict", "datasets", "data"):
        nested = payload.get(wrapper)
        if isinstance(nested, dict):
            payload = nested
            break
    return payload


def normalize_tsp_item(name, item):
    if isinstance(item, np.ndarray) and item.shape == ():
        item = item.item()
    if isinstance(item, dict):
        coords = next((item[key] for key in
                       ("coordinates", "coords", "node_coords", "node_coordinates")
                       if key in item), None)
        optimum = next((item[key] for key in
                        ("optimal_value", "optimum", "optimal", "bks")
                        if key in item), None)
        edge_type = next((item[key] for key in
                          ("edge_weight_type", "distance_type", "weight_type")
                          if key in item), "EUC_2D")
    elif isinstance(item, (tuple, list)) and len(item) >= 1:
        coords = item[0]
        optimum = item[1] if len(item) >= 2 else None
        edge_type = item[2] if len(item) >= 3 else "EUC_2D"
    else:
        return None
    if coords is None:
        return None
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        raise ValueError(f"TSP instance {name!r} has invalid coordinates shape {coords.shape}")
    return coords[:, :2], optimum, str(edge_type).upper()


def dist_matrix(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def gap_percent(value, optimum):
    """Relative excess over the best-known value, expressed as percent."""
    if value is None or optimum is None:
        return None
    try:
        optimum = float(optimum)
        return None if not math.isfinite(optimum) or optimum == 0 else (float(value) - optimum) / optimum * 100.0
    except (TypeError, ValueError):
        return None


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read object arrays written by NumPy 2.x in older NumPy environments."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _read_npy_compat(raw: bytes):
    """Read one .npy member without letting np.load choose the pickle module."""
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    elif version == (3, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported .npy version {version}")
    if dtype.hasobject:
        value = _NumpyCompatUnpickler(stream).load()
        return np.asarray(value, dtype=dtype)
    count = int(np.prod(shape, dtype=np.int64))
    values = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count)
    if fortran_order:
        return values.reshape(shape, order="F")
    return values.reshape(shape)


def load_cvrp_dict(path: Path) -> dict:
    """Load cvrp_dict from NPZ, including cross-version object-array pickles."""
    with zipfile.ZipFile(path) as archive:
        member = "cvrp_dict.npy"
        if member not in archive.namelist():
            candidates = [x for x in archive.namelist() if x.endswith(".npy")]
            raise ValueError(f"{path} has no cvrp_dict.npy; members={candidates!r}")
        value = _read_npy_compat(archive.read(member))
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        raise ValueError(f"cvrp_dict must contain a dictionary, got {type(value).__name__}")
    return value


def tsplib_matrix(coords: np.ndarray, edge_type: str) -> np.ndarray:
    """Build a TSPLIB-compatible matrix for coordinate instances."""
    d = coords[:, None, :] - coords[None, :, :]
    if edge_type == "EUC_2D":
        return np.floor(np.sqrt(np.sum(d * d, axis=2)) + 0.5)
    if edge_type == "CEIL_2D":
        return np.ceil(np.sqrt(np.sum(d * d, axis=2)))
    if edge_type == "ATT":
        rij = np.sqrt(np.sum(d * d, axis=2) / 10.0)
        tij = np.floor(rij + 0.5)
        return tij + (tij < rij)
    if edge_type == "GEO":
        def rad(x):
            deg = np.floor(x); return np.pi * (deg + 5.0 * (x - deg) / 3.0) / 180.0
        lat, lon = rad(coords[:, 0]), rad(coords[:, 1])
        q1 = np.cos(lon[:, None] - lon[None, :])
        q2 = np.cos(lat[:, None] - lat[None, :])
        q3 = np.cos(lat[:, None] + lat[None, :])
        arg = np.clip(0.5 * ((1 + q1) * q2 - (1 - q1) * q3), -1.0, 1.0)
        return np.floor(6378.388 * np.arccos(arg) + 1.0)
    return dist_matrix(coords)


def load_tsp_file(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    headers = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1); headers[k.strip().upper()] = v.strip().upper()
    n = int(headers.get("DIMENSION", "0")); typ = headers.get("EDGE_WEIGHT_TYPE", "")
    if typ == "EXPLICIT":
        fmt = headers.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
        try: start = next(i for i, x in enumerate(lines) if x.strip().upper() == "EDGE_WEIGHT_SECTION")
        except StopIteration: raise ValueError(f"Missing EDGE_WEIGHT_SECTION in {path}")
        vals = [float(v) for x in lines[start + 1:] for v in x.split()
                if x.strip().upper() != "EOF"]
        m = np.zeros((n, n)); k = 0
        if fmt == "FULL_MATRIX":
            m[:] = np.asarray(vals[:n*n]).reshape(n, n)
        else:
            diag = "DIAG" in fmt; lower = fmt.startswith("LOWER");
            for i in range(n):
                js = range(i + 1 if not diag else i, n) if not lower else range(0, i + (1 if diag else 0))
                for j in js:
                    m[i, j] = m[j, i] = vals[k]; k += 1
        return np.zeros((n, 2)), m, typ
    try: start = next(i for i, x in enumerate(lines) if x.strip().upper() == "NODE_COORD_SECTION")
    except StopIteration: raise ValueError(f"Missing NODE_COORD_SECTION in {path}")
    pts = []
    for x in lines[start + 1:]:
        if x.strip().upper() in {"EOF", "DISPLAY_DATA_SECTION", "EDGE_WEIGHT_SECTION"}: break
        q = x.split()
        if len(q) >= 3: pts.append([float(q[1]), float(q[2])])
    c = np.asarray(pts, dtype=float)
    return c, tsplib_matrix(c, typ), typ


def eval_tsp(fn, coords, matrix):
    n = len(coords); route = [0]; left = set(range(1, n)); cur = 0
    while left:
        nxt = int(fn(cur, 0, np.asarray(sorted(left), dtype=int), matrix.copy()))
        if nxt not in left: return None
        route.append(nxt); left.remove(nxt); cur = nxt
    route.append(0)
    return float(sum(matrix[a, b] for a, b in zip(route, route[1:])))


def eval_cvrp(fn, item):
    # cvrp_dict values are (capacity, node_count, coordinates, demands, BKS).
    capacity, _, coords, demands, bks = item
    coords = np.asarray(coords, dtype=float); demands = np.asarray(demands, dtype=float)
    matrix = dist_matrix(coords); n = len(coords) - 1
    left, route, cur, load = set(range(1, n + 1)), [0], 0, 0.0
    while left:
        feasible = np.asarray(sorted(x for x in left if load + demands[x] <= capacity), dtype=int)
        if feasible.size == 0:
            if cur == 0: return None
            route.append(0); cur, load = 0, 0.0; continue
        nxt = int(fn(cur, 0, feasible.copy(), float(capacity-load), demands.copy(), matrix.copy()))
        if nxt == 0:
            if cur == 0: return None
            route.append(0); cur, load = 0, 0.0; continue
        if nxt not in left or nxt not in feasible: return None
        route.append(nxt); left.remove(nxt); load += demands[nxt]; cur = nxt
    route.append(0)
    cost = float(sum(matrix[a, b] for a, b in zip(route, route[1:])))
    return cost, int(sum(x == 0 for x in route[1:-1])), float(bks) if bks is not None else None


def parse_solomon(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cap = None; rows = []; in_customer = False; expect_capacity = False
    for raw in lines:
        s = raw.strip()
        if not s: continue
        u = s.upper()
        if u.startswith("NUMBER") and "CAPACITY" in u:
            expect_capacity = True; continue
        if expect_capacity:
            nums = [x for x in s.split() if x.replace('.', '', 1).isdigit()]
            if nums:
                cap = float(nums[-1]); expect_capacity = False; continue
        if u.startswith("VEHICLE") or u.startswith("NUMBER"):
            continue
        if "CAPACITY" in u:
            nums = [x for x in s.split() if x.replace('.', '', 1).isdigit()]
            if nums: cap = float(nums[-1])
        if u.startswith("CUSTOMER"):
            in_customer = True; continue
        if not in_customer: continue
        p = s.split()
        if len(p) >= 7 and p[0].lstrip("-+").isdigit():
            rows.append([float(x) for x in p[:7]])
    if not rows: raise ValueError(f"No CUSTOMER rows in {path}")
    rows.sort(key=lambda x: int(x[0])); cap = cap or 1e30
    coords = np.asarray([[x[1], x[2]] for x in rows]); demands = np.asarray([x[3] for x in rows])
    windows = np.asarray([[x[4], x[5]] for x in rows]); service = np.asarray([x[6] for x in rows])
    return coords, dist_matrix(coords), demands, cap, service, windows


def eval_vrptw(fn, data):
    coords, matrix, demands, capacity, service, windows = data
    n = len(coords) - 1; left, route, cur, load, now = set(range(1, n+1)), [0], 0, 0.0, 0.0
    while left:
        feasible = np.asarray(sorted(x for x in left if load + demands[x] <= capacity and
            max(now + matrix[cur, x], windows[x, 0]) <= windows[x, 1]), dtype=int)
        if feasible.size == 0:
            if cur == 0: return None
            route.append(0); cur, load, now = 0, 0.0, 0.0; continue
        nxt = int(fn(cur, 0, feasible.copy(), float(capacity-load), float(now),
                     demands.copy(), matrix.copy(), windows.copy()))
        if nxt == 0:
            if cur == 0: return None
            route.append(0); cur, load, now = 0, 0.0, 0.0; continue
        if nxt not in left or nxt not in feasible: return None
        now = max(now + matrix[cur, nxt], windows[nxt, 0]) + service[nxt]
        route.append(nxt); left.remove(nxt); load += demands[nxt]; cur = nxt
    route.append(0)
    cost = float(sum(matrix[a, b] for a, b in zip(route, route[1:])))
    return int(sum(x == 0 for x in route[1:-1])), cost


def run(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a diagnostic file before loading the model/code, so early errors
    # never look like a silent no-op.
    output_path.touch()
    print(f"Starting task={args.task} logs={args.logs} data={args.data}", flush=True)
    record, code = best_program(Path(args.logs)); fn = load_function(code); rows = []
    print(f"algorithm_id={record.get('algorithm_id')} task={args.task}", flush=True)
    if args.task == "tsp":
        tsp_fields = ["instance", "node_count", "edge_weight_type",
                      "optimal_value", "score", "distance", "gap"]
        # Create the result file immediately and append each completed instance.
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            csv.DictWriter(file, fieldnames=tsp_fields).writeheader()

        def append_tsp(row):
            rows.append(row)
            with output_path.open("a", newline="", encoding="utf-8-sig") as file:
                csv.DictWriter(file, fieldnames=tsp_fields).writerow(row)

        data_path = Path(args.data)
        if data_path.suffix.lower() in {".pkl", ".pickle"}:
            with data_path.open("rb") as file:
                datasets = normalize_tsp_datasets(pickle.load(file))
            skipped = []
            for name, item in datasets.items():
                normalized = normalize_tsp_item(name, item)
                if normalized is None:
                    skipped.append((name, type(item).__name__,
                                    sorted(item.keys()) if isinstance(item, dict) else None))
                    continue
                c, optimum, typ = normalized
                if args.max_nodes > 0 and len(c) > args.max_nodes:
                    continue
                print(f"Evaluating {name} ({len(c)} nodes)...", flush=True)
                cost = eval_tsp(fn, c, tsplib_matrix(c, typ))
                append_tsp({"instance": str(name), "node_count": len(c),
                            "edge_weight_type": typ,
                            "optimal_value": optimum,
                            "score": None if cost is None else -cost,
                            "distance": cost, "gap": gap_percent(cost, optimum)})
                print(f"Completed {name}: distance={cost}", flush=True)
            if not rows:
                detail = skipped[:3]
                raise ValueError(
                    f"No recognizable TSP instances in {data_path}; "
                    f"root keys={list(datasets)[:5]!r}, examples={detail!r}")
        else:
            for p in sorted(data_path.glob("*.tsp")):
                try: c, m, typ = load_tsp_file(p)
                except (ValueError, TypeError): continue
                if len(c) > 1:
                    cost=eval_tsp(fn,c,m); rows.append({"instance":p.stem,"edge_weight_type":typ,"score":None if cost is None else -cost,"distance":cost})
    elif args.task == "cvrp":
        cvrp_fields = ["instance", "vehicles", "distance", "bks", "gap"]
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            csv.DictWriter(file, fieldnames=cvrp_fields).writeheader()
        print(f"Loading CVRP dataset: {args.data}", flush=True)
        d = load_cvrp_dict(Path(args.data))
        print(f"Loaded {len(d)} CVRP instances; output={output_path}", flush=True)
        for index, (name, item) in enumerate(d.items(), 1):
            try:
                out = eval_cvrp(fn, item)
                row = {"instance": name,
                       "vehicles": None if out is None else out[1],
                       "distance": None if out is None else out[0],
                       "bks": None if out is None else out[2],
                       "gap": None if out is None else gap_percent(out[0], out[2])}
            except Exception as exc:
                print(f"CVRP instance {name!r} failed: {exc}", flush=True)
                row = {"instance": name, "vehicles": None, "distance": None,
                       "bks": None, "gap": None}
            rows.append(row)
            with output_path.open("a", newline="", encoding="utf-8-sig") as file:
                csv.DictWriter(file, fieldnames=cvrp_fields).writerow(row)
            print(f"Completed {index}/{len(d)}: {name}", flush=True)
        print(f"completed={len(rows)} output={output_path}", flush=True)
        return
    else:
        vrptw_fields = ["instance", "vehicles", "distance"]
        # Write the header before parsing any input and append each Solomon
        # instance as soon as it finishes, so an interrupted run is usable.
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            csv.DictWriter(file, fieldnames=vrptw_fields).writeheader()
        data_path = Path(args.data)
        files = sorted(data_path.rglob("*.txt"))
        print(f"Found {len(files)} Solomon files; output={output_path}", flush=True)
        for index, p in enumerate(files, 1):
            try:
                out = eval_vrptw(fn, parse_solomon(p))
                row = {
                    "instance": p.stem,
                    "vehicles": None if out is None else out[0],
                    "distance": None if out is None else out[1],
                }
            except Exception as exc:
                print(f"VRPTW instance {p.name!r} failed: {exc}", flush=True)
                row = {"instance": p.stem, "vehicles": None, "distance": None}
            rows.append(row)
            with output_path.open("a", newline="", encoding="utf-8-sig") as file:
                csv.DictWriter(file, fieldnames=vrptw_fields).writerow(row)
            print(f"Completed {index}/{len(files)}: {p.stem}", flush=True)
        print(f"completed={len(rows)} output={output_path}", flush=True)
        return
    if args.task == "tsp":
        print(f"completed={len(rows)} output={output_path}", flush=True)
        return
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r});
    with open(args.output,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"algorithm_id={record.get('algorithm_id')} instances={len(rows)} output={args.output}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=("tsp","cvrp","vrptw"),required=True); p.add_argument("--logs",required=True); p.add_argument("--data",required=True); p.add_argument("--output",required=True); p.add_argument("--max-nodes",type=int,default=200,help="TSP maximum node count, inclusive");
    try:
        run(p.parse_args())
    except Exception:
        traceback.print_exc()
        raise
