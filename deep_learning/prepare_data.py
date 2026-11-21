"""Rebuild the processed METR-LA files from the raw data (same steps as notebooks 02 and 03).

Used on machines that only have the repository (e.g. a Kaggle GPU session):

    python -m deep_learning.prepare_data --download

``--download`` fetches the four raw files and refuses any whose SHA-256 differs from
``data/README.md``.  The committed ``splits.json`` / ``scaler.json`` / ``sensor_order.csv``
are *checked*, never overwritten, so every machine trains on exactly the Phase 1 split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request

import numpy as np

from src import config, data_loader as dl, graph_utils as gu, preprocessing as pp

DCRNN_GRAPH = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/"
RAW_FILES = {  # path -> (url, sha256) ; checksums from data/README.md
    config.METR_LA_H5: ("https://huggingface.co/datasets/MintBruce/SkyTraffic/resolve/main/metr-la.h5",
                        "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"),
    config.SENSOR_IDS_TXT: (DCRNN_GRAPH + "graph_sensor_ids.txt",
                            "3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990"),
    config.SENSOR_LOCATIONS_CSV: (DCRNN_GRAPH + "graph_sensor_locations.csv",
                                  "eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f"),
    config.DISTANCES_CSV: (DCRNN_GRAPH + "distances_la_2012.csv",
                           "a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5"),
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw() -> None:
    for path, (url, digest) in RAW_FILES.items():
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("downloading", path.name)
            urllib.request.urlretrieve(url, path)
        if sha256(path) != digest:
            path.unlink()
            raise RuntimeError(f"{path.name}: checksum mismatch, file deleted")
    print("raw files present, checksums OK")


def same(a, b) -> bool:
    """JSON equality, with floats compared to 1e-9 relative (summation order differs across CPUs)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return bool(np.isclose(a, b, rtol=1e-9, atol=0))
    return a == b


def check_or_write(path, fresh: str) -> None:
    """Write ``fresh`` if the file is missing, otherwise require the same content as the committed file."""
    if not path.is_file():
        path.write_text(fresh)
        print("wrote", path.name)
    elif not same(json.loads(path.read_text()), json.loads(fresh)):
        raise RuntimeError(f"{path.name} differs from the committed version - the split/scaler would change")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true", help="fetch and checksum the raw files first")
    args = ap.parse_args()
    config.ensure_dirs()
    if args.download:
        download_raw()
    dl.require_raw_files()

    # notebook 02: clean, split, scale
    raw = dl.load_metr_la_h5()
    speed, mask = dl.to_speed_and_mask(raw.values)
    splits = pp.make_chronological_splits(raw.timestamps)
    scaler = pp.SpeedScaler.fit(speed, mask, splits.train)
    tmp = config.DATA_PROCESSED / "_check.json"
    for obj, target in ((splits, config.SPLITS_JSON), (scaler, config.SCALER_JSON)):
        obj.to_json(tmp)
        check_or_write(target, tmp.read_text())
    tmp.unlink()
    dl.save_clean_traffic(dl.CleanTraffic(speed, mask, raw.timestamps, raw.sensor_ids))

    # notebook 03: directed adjacency in the canonical sensor order
    ordered, _ = gu.validate_sensor_alignment(raw.sensor_ids, dl.load_sensor_locations(), dl.load_sensor_ids_txt())
    adj, dist, report = gu.build_adjacency_from_distances(dl.load_distance_table(), raw.sensor_ids)
    np.save(config.ADJACENCY_NPY, adj.astype(np.float32))
    np.save(config.DISTANCE_MATRIX_NPY, dist.astype(np.float32))
    committed = gu.load_graph_artifacts(expected_sensor_ids=raw.sensor_ids)   # raises on any order mismatch
    n_edges = int(((adj > 0) & ~np.eye(len(adj), dtype=bool)).sum())
    assert n_edges == committed.meta["n_edges_directed"], (n_edges, committed.meta["n_edges_directed"])
    print(f"processed data ready: {speed.shape[0]} steps x {speed.shape[1]} sensors, "
          f"{n_edges} directed edges, sigma {report.sigma_m:.0f} m")


if __name__ == "__main__":
    main()
