"""Schema validation of the loaders and alignment of sensors / graph / traffic on the REAL files."""
import h5py
import numpy as np
import pandas as pd
import pytest

from src import config, data_loader as dl, graph_utils as gu

real_data = pytest.mark.skipif(bool(dl.missing_raw_files()), reason="raw METR-LA files not present")


def _ns(n):
    """int64 nanoseconds since epoch (pandas 3 may use a coarser unit internally)."""
    return pd.date_range("2012-03-01", periods=n, freq="5min").as_unit("ns").asi8


def _write_h5(path, values, columns, index_ns, key="df", columns_dtype="S6"):
    with h5py.File(path, "w") as f:
        g = f.create_group(key)
        g.attrs["nblocks"] = 1
        g.create_dataset("axis0", data=np.array(columns, dtype=columns_dtype))
        g.create_dataset("block0_items", data=np.array(columns, dtype=columns_dtype))
        g.create_dataset("axis1", data=np.asarray(index_ns, dtype=np.int64))
        g.create_dataset("block0_values", data=values)


def test_loader_reads_a_well_formed_file_and_orders_columns(tmp_path):
    p = tmp_path / "ok.h5"
    ns = _ns(4)
    _write_h5(p, np.arange(8, dtype=float).reshape(4, 2), ["222", "111"], ns)
    raw = dl.load_metr_la_h5(p)
    assert raw.sensor_ids == ["222", "111"] and raw.values.shape == (4, 2)
    assert raw.timestamps[1] - raw.timestamps[0] == pd.Timedelta("5min")


def test_loader_rejects_wrong_key_shape_and_unsorted_time(tmp_path):
    ns = _ns(4)
    _write_h5(tmp_path / "a.h5", np.zeros((4, 2)), ["1", "2"], ns, key="other")
    with pytest.raises(dl.DataSchemaError, match="Key"):
        dl.load_metr_la_h5(tmp_path / "a.h5")
    _write_h5(tmp_path / "b.h5", np.zeros((3, 2)), ["1", "2"], ns)
    with pytest.raises(dl.DataSchemaError, match="inconsistent"):
        dl.load_metr_la_h5(tmp_path / "b.h5")
    _write_h5(tmp_path / "c.h5", np.zeros((4, 2)), ["1", "2"], ns[::-1].copy())
    with pytest.raises(dl.DataSchemaError, match="increasing"):
        dl.load_metr_la_h5(tmp_path / "c.h5")


def test_loader_rejects_duplicate_sensor_ids_and_missing_file(tmp_path):
    ns = _ns(4)
    _write_h5(tmp_path / "d.h5", np.zeros((4, 2)), ["1", "1"], ns)
    with pytest.raises(dl.DataSchemaError, match="Duplicate"):
        dl.load_metr_la_h5(tmp_path / "d.h5")
    with pytest.raises(FileNotFoundError):
        dl.load_metr_la_h5(tmp_path / "nope.h5")


def test_zero_is_missing_not_zero_speed():
    values = np.array([[0.0, 55.0], [30.0, 0.0]])
    speed, mask = dl.to_speed_and_mask(values)
    assert np.isnan(speed[0, 0]) and np.isnan(speed[1, 1])
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])
    assert speed.dtype == np.float32
    with pytest.raises(dl.DataSchemaError):
        dl.to_speed_and_mask(np.array([[-1.0]]))


def test_clean_traffic_roundtrip(tmp_path):
    ts = pd.date_range("2012-03-01", periods=3, freq="5min")
    clean = dl.CleanTraffic(np.array([[1.0, np.nan]] * 3, np.float32), np.array([[True, False]] * 3), ts, ["7", "8"])
    p = tmp_path / "c.npz"; dl.save_clean_traffic(clean, p)
    back = dl.load_clean_traffic(p)
    assert back.sensor_ids == ["7", "8"] and (back.timestamps == ts).all() and back.mask.dtype == bool


@real_data
def test_real_files_have_the_documented_shape_and_frequency():
    raw = dl.load_metr_la_h5()
    assert raw.values.shape == (34272, 207)
    assert len(set(raw.sensor_ids)) == 207
    steps = np.diff(raw.timestamps.values).astype("timedelta64[s]").astype(int)
    assert (steps == 300).all()
    assert not np.isnan(raw.values).any()


@real_data
def test_real_sensor_ids_align_across_traffic_locations_and_graph():
    raw = dl.load_metr_la_h5()
    ordered, rep = gu.validate_sensor_alignment(raw.sensor_ids, dl.load_sensor_locations(), dl.load_sensor_ids_txt())
    assert list(ordered["sensor_id"]) == raw.sensor_ids
    assert rep["graph_order_equals_traffic_order"]
    assert not rep["traffic_without_location"] and not rep["location_without_traffic"]


@real_data
def test_real_adjacency_is_square_directed_and_row_aligned():
    raw = dl.load_metr_la_h5()
    adj, dist, rep = gu.build_adjacency_from_distances(dl.load_distance_table(), raw.sensor_ids)
    assert adj.shape == (207, 207)
    stats = gu.graph_stats(adj)
    assert stats["n_self_loops"] == 207 and not stats["is_symmetric"]
    assert stats["n_nonzero_total"] == 1722          # the edge count reported for DCRNN's METR-LA graph
    assert 0 < adj[adj > 0].min() and adj.max() <= 1.0
    assert rep.n_duplicate_pairs == 0
