"""Adjacency construction, graph statistics, sensor alignment and ordering guards."""
import numpy as np
import pandas as pd
import pytest

from src import graph_utils as gu


def _distance_table():
    # directed road distances in metres; sensors A,B,C,D (+ one unknown sensor Z)
    rows = []
    ids = ["A", "B", "C", "D"]
    for i in ids:
        rows.append((i, i, 0.0))
    rows += [("A", "B", 1000.0), ("B", "A", 2500.0), ("B", "C", 1500.0), ("C", "D", 900.0),
             ("A", "D", 20000.0), ("Z", "A", 500.0), ("A", "Z", 500.0)]
    return pd.DataFrame(rows, columns=["from", "to", "cost"]), ids


def test_adjacency_shape_diagonal_and_weight_formula():
    df, ids = _distance_table()
    adj, dist, rep = gu.build_adjacency_from_distances(df, ids, normalized_k=0.1)
    assert adj.shape == (4, 4) == dist.shape
    assert np.all(np.diag(adj) == 1.0)                       # d_ii = 0 -> weight 1 (self loop)
    sigma = rep.sigma_m
    assert adj[0, 1] == pytest.approx(np.exp(-(1000.0 / sigma) ** 2), rel=1e-5)
    assert adj.min() >= 0 and adj.max() <= 1


def test_threshold_removes_far_pairs_and_unlisted_pairs_have_no_edge():
    df, ids = _distance_table()
    adj, _, _ = gu.build_adjacency_from_distances(df, ids, normalized_k=0.1)
    assert adj[0, 3] == 0.0                                  # 20 km pair is below the threshold
    assert adj[3, 0] == 0.0                                  # never listed -> inf distance -> no edge
    assert np.all((adj == 0) | (adj >= 0.1))


def test_graph_is_directed_not_symmetrised():
    df, ids = _distance_table()
    adj, _, _ = gu.build_adjacency_from_distances(df, ids, normalized_k=0.01)
    assert adj[0, 1] != adj[1, 0]                            # 1000 m vs 2500 m
    assert adj[2, 3] > 0 and adj[3, 2] == 0                  # one-way pair stays one-way
    assert not gu.graph_stats(adj)["is_symmetric"]


def test_unknown_sensor_rows_are_ignored_and_counted():
    df, ids = _distance_table()
    _, _, rep = gu.build_adjacency_from_distances(df, ids)
    assert rep.n_rows_skipped_unknown_sensor == 2
    assert rep.n_rows_used + rep.n_rows_skipped_unknown_sensor == rep.n_distance_rows


def test_duplicate_pairs_are_reported_and_resolved_deterministically():
    df, ids = _distance_table()
    df = pd.concat([df, pd.DataFrame([("A", "B", 5000.0)], columns=df.columns)])
    adj, dist, rep = gu.build_adjacency_from_distances(df, ids)
    assert rep.n_duplicate_pairs == 1 and dist[0, 1] == 1000.0       # smaller cost kept


def test_duplicate_sensor_ids_are_rejected():
    df, _ = _distance_table()
    with pytest.raises(gu.AlignmentError):
        gu.build_adjacency_from_distances(df, ["A", "B", "B"])


def test_stats_isolated_nodes_reciprocity_and_components():
    adj = np.eye(5, dtype=np.float32)
    adj[0, 1] = adj[1, 0] = 0.5          # reciprocal pair
    adj[1, 2] = 0.4                       # one-way
    stats = gu.graph_stats(adj)
    assert stats["n_edges_directed"] == 3 and stats["n_reciprocal_edges"] == 2
    assert stats["isolated_nodes"] == [3, 4]
    assert stats["n_weak_components"] == 3
    assert stats["n_self_loops"] == 5
    assert list(stats["in_degree"]) == [1, 1, 1, 0, 0] and list(stats["out_degree"]) == [1, 2, 0, 0, 0]


def test_edge_index_convention_and_validity():
    adj = np.zeros((4, 4), np.float32); adj[0, 1] = 0.7; adj[2, 3] = 0.3; np.fill_diagonal(adj, 1)
    ei, w = gu.adjacency_to_edge_index(adj)
    assert ei.shape == (2, 2) and ei.dtype == np.int64
    assert {tuple(e) for e in ei.T} == {(0, 1), (2, 3)}           # (source, target)
    assert ei.min() >= 0 and ei.max() < 4 and np.all(ei[0] != ei[1])
    ei2, _ = gu.adjacency_to_edge_index(adj, add_self_loops=True)
    assert ei2.shape[1] == 2 + 4


def test_attention_masks_follow_edge_direction_and_include_self():
    adj = np.zeros((3, 3), np.float32); adj[0, 1] = 1.0      # edge 0 -> 1
    m = gu.neighbor_masks(adj)
    assert m["in"][1, 0] and not m["in"][0, 1]               # node 1 listens to its upstream node 0
    assert m["out"][0, 1] and not m["out"][1, 0]
    for k in m.values():
        assert k.diagonal().all()
    assert m["self"].sum() == 3


def test_sensor_order_hash_detects_reordering():
    ids = ["1", "2", "3"]
    h = gu.sensor_order_hash(ids)
    gu.assert_sensor_order(ids, h)
    with pytest.raises(gu.AlignmentError):
        gu.assert_sensor_order(["2", "1", "3"], h)


def _locs(ids, lat0=34.0):
    return pd.DataFrame({"sensor_id": ids, "latitude": np.arange(len(ids)) + lat0, "longitude": -118.0})


def test_alignment_matches_by_id_not_row_position():
    traffic = ["10", "20", "30"]
    locs = _locs(["30", "10", "20"])                          # shuffled rows
    ordered, rep = gu.validate_sensor_alignment(traffic, locs, traffic)
    assert list(ordered["sensor_id"]) == traffic
    assert ordered.loc[ordered.sensor_id == "30", "latitude"].item() == 34.0   # row 0 of the shuffled file
    assert not rep["location_row_order_equals_traffic_order"]


def test_alignment_reports_missing_unknown_and_duplicate_ids():
    traffic = ["10", "20", "30"]
    with pytest.raises(gu.AlignmentError, match="traffic_without_location"):
        gu.validate_sensor_alignment(traffic, _locs(["10", "20"]), traffic)
    with pytest.raises(gu.AlignmentError, match="graph_not_in_traffic"):
        gu.validate_sensor_alignment(traffic, _locs(traffic), traffic + ["99"])
    with pytest.raises(gu.AlignmentError, match="duplicate_location_ids"):
        gu.validate_sensor_alignment(traffic, _locs(["10", "20", "30", "30"]), traffic)


def test_graph_artifact_roundtrip_guards_order(tmp_path, monkeypatch):
    from src import config
    for name in ("DATA_PROCESSED", "ADJACENCY_NPY", "DISTANCE_MATRIX_NPY", "SENSOR_ORDER_CSV", "GRAPH_META_JSON"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).name if name != "DATA_PROCESSED" else tmp_path)
    ids = ["a", "b", "c"]
    adj = np.eye(3, dtype=np.float32); dist = np.zeros((3, 3), np.float32)
    ordered = _locs(ids); ordered.insert(0, "node", range(3))
    gu.save_graph_artifacts(adj, dist, ordered, {"normalized_k": 0.1})
    art = gu.load_graph_artifacts(expected_sensor_ids=ids)
    assert art.adjacency.shape == (3, 3) and art.sensor_ids == ids
    with pytest.raises(gu.AlignmentError):
        gu.load_graph_artifacts(expected_sensor_ids=["b", "a", "c"])
    with pytest.raises(gu.AlignmentError):
        gu.save_graph_artifacts(np.eye(2, dtype=np.float32), dist, ordered, {})


def test_haversine_known_distance_and_symmetry():
    # one degree of latitude is about 111.2 km
    d = gu.haversine_matrix(np.array([34.0, 35.0]), np.array([-118.0, -118.0]))
    assert d[0, 1] == pytest.approx(111_195, rel=1e-3) and d[0, 1] == d[1, 0] and d[0, 0] == 0
