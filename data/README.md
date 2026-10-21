# Data

Nothing in this folder is committed to Git (see `.gitignore`); the files are reproducible from the sources below.

## Required raw files (`data/raw/`)

| File | Content | Source | Size |
|---|---|---|---|
| `metr-la.h5` | METR-LA traffic speeds, 34,272 five-minute steps x 207 sensors (1 Mar - 27 Jun 2012), pandas `HDFStore`, key `df` | The original is linked from the [DCRNN README](https://github.com/liyaguang/DCRNN) (Google Drive `10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX`, Baidu Yun). The copy used for the reported results was downloaded from the third-party mirror `https://huggingface.co/datasets/MintBruce/SkyTraffic/resolve/main/metr-la.h5` | 57,038,056 bytes |
| `sensor_graph/graph_sensor_ids.txt` | the 207 sensor ids, comma separated, in the DCRNN order | `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt` | 1,448 bytes |
| `sensor_graph/graph_sensor_locations.csv` | `index, sensor_id, latitude, longitude` | `.../data/sensor_graph/graph_sensor_locations.csv` | 6,341 bytes |
| `sensor_graph/distances_la_2012.csv` | directed road-network distances `from, to, cost` (metres) for 4,106 LA sensors | `.../data/sensor_graph/distances_la_2012.csv` | 6,393,348 bytes |

SHA-256 of the files used for the reported results:

```
64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05  metr-la.h5
3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990  sensor_graph/graph_sensor_ids.txt
eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f  sensor_graph/graph_sensor_locations.csv
a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5  sensor_graph/distances_la_2012.csv
```

Download example (run from the project root):

```bash
mkdir -p data/raw/sensor_graph
B=https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph
for f in graph_sensor_ids.txt graph_sensor_locations.csv distances_la_2012.csv; do curl -fsSL -o data/raw/sensor_graph/$f $B/$f; done
curl -fSL -o data/raw/metr-la.h5 "https://huggingface.co/datasets/MintBruce/SkyTraffic/resolve/main/metr-la.h5"
shasum -a 256 data/raw/metr-la.h5      # compare with the checksum above
```

The HDF5 file is read with `h5py` (no unpickling). The shipped `adj_mx.pkl` of DCRNN is intentionally **not** used (pickle can execute code); the adjacency is rebuilt from `distances_la_2012.csv`.

## Data semantics

* Speeds are in **mph**, five-minute averages. A reading of exactly `0.0` means **missing** (evidence in notebook 01); it is converted to NaN plus an `observed` mask and is never treated as a zero speed.
* The file has no vehicle-flow or occupancy channel; none is used or invented.
* Timestamps are naive; Pacific local time is assumed (consistent with the commute pattern in notebook 01).

## Generated files (`data/processed/`)

| File | Created by | Content |
|---|---|---|
| `traffic_clean.npz` | notebook 02 | speed (NaN = missing), observed mask, timestamps, canonical sensor ids |
| `splits.json` | notebook 02 | exact chronological split boundaries (indices and timestamps) |
| `scaler.json` | notebook 02 | training-only mean / std and per-sensor training means |
| `adjacency.npy`, `road_distance_matrix.npy` | notebook 03 | validated directed weighted adjacency / road distances in canonical sensor order |
| `sensor_order.csv`, `graph_meta.json` | notebook 03 | canonical sensor order with coordinates and its fingerprint (guards against re-ordering) |
