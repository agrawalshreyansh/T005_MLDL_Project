"""Loading and schema validation of the METR-LA files.

The HDF5 file is read with ``h5py`` rather than ``pandas.read_hdf``.  pandas'
fixed-format reader can unpickle object arrays stored inside the file, which
is an unnecessary risk for a third-party mirror.  For METR-LA the sensor ids
are fixed-width byte strings and the timestamps are int64 nanoseconds, so no
pickled objects are needed; the loader asserts this.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import config


class DataSchemaError(RuntimeError):
    """Raised when a data file is missing or does not have the expected structure."""


# --------------------------------------------------------------------------- #
# Availability checks
# --------------------------------------------------------------------------- #
def missing_raw_files() -> dict[Path, str]:
    """Return ``{path: description}`` for every required raw file that is absent."""
    return {p: d for p, d in config.REQUIRED_RAW_FILES.items() if not p.is_file()}


def require_raw_files() -> None:
    """Raise a precise, actionable error if any required raw file is absent."""
    missing = missing_raw_files()
    if missing:
        lines = [f"  - {p.relative_to(config.PROJECT_ROOT)}: {d}" for p, d in missing.items()]
        raise FileNotFoundError(
            "Missing required data files:\n" + "\n".join(lines)
            + "\nSee data/README.md for download instructions."
        )


# --------------------------------------------------------------------------- #
# HDF5 inspection and loading
# --------------------------------------------------------------------------- #
def inspect_h5(path: Path = config.METR_LA_H5) -> dict:
    """Describe the HDF5 layout (groups, datasets, shapes, dtypes, attributes)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    report: dict = {"root_keys": [], "datasets": {}, "group_attrs": {}}
    with h5py.File(path, "r") as handle:
        report["root_keys"] = list(handle.keys())

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                report["datasets"][name] = {"shape": tuple(obj.shape), "dtype": str(obj.dtype)}
            else:
                report["group_attrs"][name] = {
                    k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in obj.attrs.items()
                }

        handle.visititems(visit)
    return report


@dataclass(frozen=True)
class RawTraffic:
    """Raw METR-LA content exactly as stored (zeros are NOT yet interpreted)."""

    values: np.ndarray          # float64 [T, N], 0.0 marks a missing reading
    timestamps: pd.DatetimeIndex  # length T, naive local time
    sensor_ids: list[str]       # length N, canonical sensor order


def load_metr_la_h5(path: Path = config.METR_LA_H5, key: str = config.H5_KEY) -> RawTraffic:
    """Load METR-LA from a pandas-fixed-format HDF5 file with strict validation."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}. See data/README.md.")
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise DataSchemaError(f"Key {key!r} not in {path}; found {list(handle.keys())}")
        group = handle[key]
        needed = ("axis0", "axis1", "block0_values", "block0_items")
        absent = [n for n in needed if n not in group]
        if absent:
            raise DataSchemaError(f"HDF5 group {key!r} lacks datasets {absent}; has {list(group.keys())}")
        if group.attrs.get("nblocks", 1) != 1:
            raise DataSchemaError("Expected a single homogeneous block of speeds.")
        for name in needed:
            if group[name].dtype.kind == "O":
                raise DataSchemaError(
                    f"Dataset {name} has object dtype (would require unpickling); refusing to load."
                )
        values = group["block0_values"][:]
        columns = group["axis0"][:]
        items = group["block0_items"][:]
        index_ns = group["axis1"][:]

    if values.ndim != 2:
        raise DataSchemaError(f"Expected a 2-D value block, got shape {values.shape}")
    if values.shape != (len(index_ns), len(columns)):
        # pandas stores blocks as [columns, rows] in some versions; this file is [rows, columns]
        raise DataSchemaError(
            f"Value block {values.shape} inconsistent with index ({len(index_ns)}) x columns ({len(columns)})"
        )
    if not np.array_equal(columns, items):
        raise DataSchemaError("axis0 (column labels) and block0_items disagree.")
    if index_ns.dtype != np.int64:
        raise DataSchemaError(f"Timestamp axis must be int64 nanoseconds, got {index_ns.dtype}")

    timestamps = pd.DatetimeIndex(pd.to_datetime(index_ns, unit="ns"))
    sensor_ids = [c.decode("ascii") for c in columns]
    if len(set(sensor_ids)) != len(sensor_ids):
        raise DataSchemaError("Duplicate sensor ids in the traffic file.")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise DataSchemaError("Timestamps are not strictly increasing.")
    return RawTraffic(values=values.astype(np.float64), timestamps=timestamps, sensor_ids=sensor_ids)


def to_speed_and_mask(
    values: np.ndarray, sentinel: float = config.MISSING_SENTINEL
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw values to ``(speed_with_nan, observed_mask)``.

    A reading equal to ``sentinel`` is *missing*; it is returned as NaN (never as a
    zero speed) and flagged ``False`` in the mask.  NaN/inf in the input are also
    treated as missing, and negative speeds raise because they are impossible.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < 0):
        raise DataSchemaError("Negative speeds found; refusing to guess their meaning.")
    observed = np.isfinite(values) & (values != sentinel)
    speed = np.where(observed, values, np.nan).astype(np.float32)
    return speed, observed


# --------------------------------------------------------------------------- #
# Graph-side files
# --------------------------------------------------------------------------- #
def load_sensor_ids_txt(path: Path = config.SENSOR_IDS_TXT) -> list[str]:
    """Read the comma-separated sensor id list shipped with DCRNN."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Sensor id list not found: {path}")
    ids = [tok.strip() for tok in Path(path).read_text().strip().split(",") if tok.strip()]
    if not ids:
        raise DataSchemaError(f"{path} contains no sensor ids")
    return ids


def load_sensor_locations(path: Path = config.SENSOR_LOCATIONS_CSV) -> pd.DataFrame:
    """Read sensor coordinates; ``sensor_id`` is returned as ``str``."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Sensor location file not found: {path}")
    df = pd.read_csv(path)
    expected = {"sensor_id", "latitude", "longitude"}
    if not expected.issubset(df.columns):
        raise DataSchemaError(f"{path} must contain {sorted(expected)}, has {list(df.columns)}")
    df = df.copy()
    df["sensor_id"] = df["sensor_id"].astype(str)
    return df


def load_distance_table(path: Path = config.DISTANCES_CSV) -> pd.DataFrame:
    """Read the directed road-network distance table (``from``, ``to``, ``cost`` in metres)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Distance table not found: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != ["from", "to", "cost"]:
        raise DataSchemaError(f"{path} must have columns ['from','to','cost'], has {list(df.columns)}")
    df = df.copy()
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    return df


# --------------------------------------------------------------------------- #
# Processed-data persistence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CleanTraffic:
    """Interpreted traffic data shared by every modelling notebook."""

    speed: np.ndarray             # float32 [T, N], NaN where missing
    mask: np.ndarray              # bool    [T, N], True where observed
    timestamps: pd.DatetimeIndex  # length T
    sensor_ids: list[str]         # length N


def save_clean_traffic(data: CleanTraffic, path: Path = config.TRAFFIC_CLEAN_NPZ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        speed=data.speed,
        mask=data.mask,
        timestamps_ns=data.timestamps.values.astype("datetime64[ns]").astype(np.int64),
        sensor_ids=np.array(data.sensor_ids, dtype="U16"),
    )


def load_clean_traffic(path: Path = config.TRAFFIC_CLEAN_NPZ) -> CleanTraffic:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found - run notebook 02_data_preprocessing first.")
    with np.load(path, allow_pickle=False) as z:
        return CleanTraffic(
            speed=z["speed"],
            mask=z["mask"],
            timestamps=pd.DatetimeIndex(pd.to_datetime(z["timestamps_ns"], unit="ns")),
            sensor_ids=[str(s) for s in z["sensor_ids"]],
        )


def load_clean_from_raw() -> CleanTraffic:
    """Convenience: raw HDF5 -> CleanTraffic without touching the processed cache."""
    raw = load_metr_la_h5()
    speed, mask = to_speed_and_mask(raw.values)
    return CleanTraffic(speed=speed, mask=mask, timestamps=raw.timestamps, sensor_ids=raw.sensor_ids)
