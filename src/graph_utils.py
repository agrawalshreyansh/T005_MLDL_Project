"""Road-graph construction, validation and persistence.

Terminology (kept strictly separate, see notebook 03):

* **coordinates** - latitude/longitude of a sensor (``graph_sensor_locations.csv``).
  They place sensors on a map; they say nothing about road connectivity.
* **road-network distance** - directed driving distance in metres from sensor *i*
  to sensor *j* (``distances_la_2012.csv``, pre-computed by the DCRNN authors).
* **weighted adjacency** - ``A[i, j] = exp(-(d_ij / sigma)^2)`` set to 0 when below
  ``normalized_k``.  This is the thresholded Gaussian kernel of DCRNN
  (``scripts/gen_adj_mx.py``).  ``A[i, j] > 0`` means a directed edge i -> j.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from . import config


class AlignmentError(RuntimeError):
    """Sensor ids of the traffic data, locations and graph disagree."""


# --------------------------------------------------------------------------- #
# Sensor ordering and alignment
# --------------------------------------------------------------------------- #
def sensor_order_hash(sensor_ids: list[str]) -> str:
    """Short fingerprint of an ordered sensor-id list (order sensitive)."""
    return hashlib.sha256("\n".join(map(str, sensor_ids)).encode()).hexdigest()[:16]


def assert_sensor_order(sensor_ids: list[str], expected_hash: str, where: str = "") -> None:
    """Fail loudly if the sensor ordering differs from the canonical one."""
    got = sensor_order_hash(sensor_ids)
    if got != expected_hash:
        raise AlignmentError(
            f"Sensor order changed{' in ' + where if where else ''}: hash {got} != canonical {expected_hash}"
        )


def validate_sensor_alignment(
    traffic_ids: list[str], locations: pd.DataFrame, graph_ids: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Match locations to the traffic columns **by id** and report every discrepancy.

    Returns ``(locations_in_canonical_order, report)``.  The canonical order is the
    column order of the traffic file.  Raises :class:`AlignmentError` when a modeled
    sensor has no location, when ids are duplicated, or when the graph id list is not
    exactly the traffic id list in the same order.
    """
    report: dict = {}
    report["n_traffic"] = len(traffic_ids)
    report["n_locations"] = int(len(locations))
    report["n_graph_ids"] = len(graph_ids)
    report["duplicate_traffic_ids"] = sorted({s for s in traffic_ids if traffic_ids.count(s) > 1})
    report["duplicate_location_ids"] = sorted(
        locations.loc[locations["sensor_id"].duplicated(keep=False), "sensor_id"].unique()
    )
    report["duplicate_graph_ids"] = sorted({s for s in graph_ids if graph_ids.count(s) > 1})
    traffic_set, loc_set, graph_set = set(traffic_ids), set(locations["sensor_id"]), set(graph_ids)
    report["traffic_without_location"] = sorted(traffic_set - loc_set)
    report["location_without_traffic"] = sorted(loc_set - traffic_set)
    report["traffic_not_in_graph"] = sorted(traffic_set - graph_set)
    report["graph_not_in_traffic"] = sorted(graph_set - traffic_set)
    report["graph_order_equals_traffic_order"] = list(graph_ids) == list(traffic_ids)
    report["location_row_order_equals_traffic_order"] = list(locations["sensor_id"]) == list(traffic_ids)

    problems = [
        k for k in (
            "duplicate_traffic_ids", "duplicate_location_ids", "duplicate_graph_ids",
            "traffic_without_location", "traffic_not_in_graph", "graph_not_in_traffic",
        ) if report[k]
    ]
    if not report["graph_order_equals_traffic_order"] and not problems:
        # same set, different order: resolvable by id, but it must be visible
        report["note"] = "graph id order differs from traffic order; the adjacency is re-indexed by id"
    if problems:
        raise AlignmentError(f"Sensor alignment failed on {problems}: " + json.dumps(
            {k: report[k] for k in problems}, default=str)[:600])

    ordered = (
        locations.drop_duplicates("sensor_id")
        .set_index("sensor_id").loc[traffic_ids].reset_index()
    )
    ordered.insert(0, "node", np.arange(len(ordered)))
    assert list(ordered["sensor_id"]) == list(traffic_ids)
    return ordered, report


# --------------------------------------------------------------------------- #
# Adjacency construction (DCRNN gen_adj_mx logic)
# --------------------------------------------------------------------------- #
@dataclass
class GraphBuildReport:
    n_nodes: int
    n_distance_rows: int
    n_rows_used: int
    n_rows_skipped_unknown_sensor: int
    n_duplicate_pairs: int
    n_finite_pairs: int
    sigma_m: float
    normalized_k: float


def build_adjacency_from_distances(
    distance_df: pd.DataFrame, sensor_ids: list[str], normalized_k: float = config.GRAPH_NORMALIZED_K
) -> tuple[np.ndarray, np.ndarray, GraphBuildReport]:
    """Thresholded Gaussian-kernel adjacency from directed road distances.

    Mirrors ``get_adjacency_matrix`` in DCRNN's ``scripts/gen_adj_mx.py``:

    1. ``dist[i, j]`` = road distance sensor i -> sensor j (``inf`` when not listed).
    2. ``sigma`` = standard deviation of all finite entries (diagonal zeros included,
       exactly as in the original code).
    3. ``A = exp(-(dist / sigma)^2)`` and entries ``< normalized_k`` are set to 0.

    Rows referring to sensors outside ``sensor_ids`` are ignored (and counted).  Rows
    are **not** symmetrised: the graph is directed.
    """
    if len(set(sensor_ids)) != len(sensor_ids):
        raise AlignmentError("Duplicate sensor ids passed to build_adjacency_from_distances")
    index = {s: i for i, s in enumerate(sensor_ids)}
    known = distance_df["from"].isin(index) & distance_df["to"].isin(index)
    used = distance_df[known]
    dup = int(used.duplicated(["from", "to"]).sum())
    used = used.groupby(["from", "to"], as_index=False)["cost"].min()  # deterministic if duplicated

    n = len(sensor_ids)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    rows = used["from"].map(index).to_numpy()
    cols = used["to"].map(index).to_numpy()
    dist[rows, cols] = used["cost"].to_numpy(dtype=np.float32)

    finite = dist[np.isfinite(dist)].flatten()
    if finite.size == 0:
        raise ValueError("No finite distances among the requested sensors.")
    sigma = finite.std()
    with np.errstate(over="ignore"):
        adj = np.exp(-np.square(dist / sigma)).astype(np.float32)
    adj[adj < normalized_k] = 0.0
    report = GraphBuildReport(
        n_nodes=n,
        n_distance_rows=int(len(distance_df)),
        n_rows_used=int(len(used)),
        n_rows_skipped_unknown_sensor=int((~known).sum()),
        n_duplicate_pairs=dup,
        n_finite_pairs=int(finite.size),
        sigma_m=float(sigma),
        normalized_k=float(normalized_k),
    )
    return adj, dist, report


# --------------------------------------------------------------------------- #
# Graph statistics
# --------------------------------------------------------------------------- #
def graph_stats(adj: np.ndarray) -> dict:
    """Structural statistics of a (possibly directed, weighted) adjacency matrix."""
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adj.shape}")
    n = adj.shape[0]
    offdiag = adj.copy()
    np.fill_diagonal(offdiag, 0.0)
    binary = offdiag > 0
    in_deg = binary.sum(axis=0)
    out_deg = binary.sum(axis=1)
    reciprocal = binary & binary.T
    n_edges = int(binary.sum())
    weights = offdiag[binary]
    n_weak, weak_labels = connected_components(sparse.csr_matrix(binary), directed=True, connection="weak")
    n_strong, _ = connected_components(sparse.csr_matrix(binary), directed=True, connection="strong")
    return {
        "n_nodes": n,
        "shape": tuple(adj.shape),
        "n_nonzero_total": int((adj > 0).sum()),
        "n_self_loops": int((np.diag(adj) > 0).sum()),
        "n_edges_directed": n_edges,
        "edge_density": n_edges / (n * (n - 1)) if n > 1 else 0.0,
        "weight_min": float(weights.min()) if n_edges else 0.0,
        "weight_max": float(weights.max()) if n_edges else 0.0,
        "weight_mean": float(weights.mean()) if n_edges else 0.0,
        "n_reciprocal_edges": int(reciprocal.sum()),
        "reciprocal_fraction": float(reciprocal.sum() / n_edges) if n_edges else 0.0,
        "is_symmetric": bool(np.allclose(adj, adj.T)),
        "max_asymmetry": float(np.abs(adj - adj.T).max()),
        "in_degree": in_deg,
        "out_degree": out_deg,
        "mean_degree": float(out_deg.mean()),
        "max_in_degree": int(in_deg.max()),
        "max_out_degree": int(out_deg.max()),
        "isolated_nodes": [int(i) for i in np.where((in_deg == 0) & (out_deg == 0))[0]],
        "n_weak_components": int(n_weak),
        "n_strong_components": int(n_strong),
        "largest_weak_component": int(np.bincount(weak_labels).max()),
    }


def adjacency_to_edge_index(adj: np.ndarray, add_self_loops: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """COO edge list ``(edge_index [2, E], edge_weight [E])`` with ``source -> target``.

    ``adj[i, j] > 0`` yields the edge ``i -> j`` (``edge_index[0] = i``, ``edge_index[1] = j``),
    which is the PyTorch-Geometric convention (messages flow source -> target).
    """
    work = np.asarray(adj, dtype=np.float32).copy()
    if add_self_loops:
        np.fill_diagonal(work, np.maximum(np.diag(work), 1.0))
    else:
        np.fill_diagonal(work, 0.0)
    src, dst = np.nonzero(work > 0)
    return np.vstack([src, dst]).astype(np.int64), work[src, dst]


def neighbor_masks(adj: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean attention masks for dense graph attention.

    ``mask[i, j] = True`` means *node i may attend to node j*.

    * ``"in"``  : j -> i is an edge (i listens to its upstream neighbours) + self.
    * ``"out"`` : i -> j is an edge (i listens to its downstream neighbours) + self.
    * ``"self"``: only the diagonal (used for the no-graph ablation).
    """
    binary = np.asarray(adj) > 0
    n = binary.shape[0]
    eye = np.eye(n, dtype=bool)
    off = binary & ~eye
    return {"in": off.T | eye, "out": off | eye, "self": eye}


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres between every pair of coordinates (``[N, N]``).

    This is *straight-line* distance over the Earth's surface: a drawing/diagnostic aid,
    **not** a road distance and not evidence that two sensors are connected.
    """
    r = 6_371_000.0
    phi, lam = np.deg2rad(np.asarray(lat, dtype=np.float64)), np.deg2rad(np.asarray(lon, dtype=np.float64))
    dphi, dlam = phi[:, None] - phi[None, :], lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
@dataclass
class GraphArtifacts:
    adjacency: np.ndarray
    distance: np.ndarray
    sensor_ids: list[str]
    locations: pd.DataFrame
    meta: dict


def save_graph_artifacts(
    adj: np.ndarray, dist: np.ndarray, locations_ordered: pd.DataFrame, meta: dict
) -> None:
    sensor_ids = list(locations_ordered["sensor_id"])
    if adj.shape != (len(sensor_ids), len(sensor_ids)):
        raise AlignmentError(f"Adjacency {adj.shape} does not match {len(sensor_ids)} sensors")
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(config.ADJACENCY_NPY, adj.astype(np.float32))
    np.save(config.DISTANCE_MATRIX_NPY, dist.astype(np.float32))
    locations_ordered.to_csv(config.SENSOR_ORDER_CSV, index=False)
    meta = dict(meta, sensor_order_hash=sensor_order_hash(sensor_ids), n_nodes=len(sensor_ids))
    config.GRAPH_META_JSON.write_text(json.dumps(meta, indent=2, default=str))


def load_graph_artifacts(expected_sensor_ids: list[str] | None = None) -> GraphArtifacts:
    """Load the validated graph; optionally assert it matches a traffic sensor order."""
    for p in (config.ADJACENCY_NPY, config.SENSOR_ORDER_CSV, config.GRAPH_META_JSON):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{p} not found - run notebook 03_graph_construction first.")
    adj = np.load(config.ADJACENCY_NPY)
    dist = np.load(config.DISTANCE_MATRIX_NPY) if config.DISTANCE_MATRIX_NPY.is_file() else np.empty((0, 0))
    locations = pd.read_csv(config.SENSOR_ORDER_CSV, dtype={"sensor_id": str})
    meta = json.loads(config.GRAPH_META_JSON.read_text())
    sensor_ids = list(locations["sensor_id"])
    assert_sensor_order(sensor_ids, meta["sensor_order_hash"], "saved graph artifacts")
    if adj.shape != (len(sensor_ids),) * 2:
        raise AlignmentError(f"Saved adjacency {adj.shape} vs {len(sensor_ids)} sensors")
    if expected_sensor_ids is not None:
        assert_sensor_order(expected_sensor_ids, meta["sensor_order_hash"], "traffic data vs saved graph")
    return GraphArtifacts(adj, dist, sensor_ids, locations, meta)


def report_to_dict(report: GraphBuildReport) -> dict:
    return asdict(report)
