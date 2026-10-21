"""Optional SUMO experiment: fixed-time vs queue-aware control of ONE simulated intersection.

Nothing here touches real traffic infrastructure; it only drives a local SUMO simulation.

STATUS: written for SUMO >= 1.18 + TraCI, but **not executed in the environment this project was developed in**
(SUMO was not installed).  The pure-Python signal logic (``signal_policies.py``) is unit-tested; the SUMO glue
code below (network conversion, TraCI loop, metrics) has not been run and may need small adjustments.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import signal_policies as sp

SIM_DIR = Path(__file__).resolve().parent
CFG = SIM_DIR / "sumo_config"
OUT = SIM_DIR / "output"
APPROACHES = {"N": "N2C", "S": "S2C", "E": "E2C", "W": "W2C"}
GROUP = {"N": sp.NS, "S": sp.NS, "E": sp.EW, "W": sp.EW}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


# --------------------------------------------------------------------------- #
# Network description (4-way intersection, one lane per approach, through movements only)
# --------------------------------------------------------------------------- #
def write_network_files(arm_length_m: float = 250.0, speed_mps: float = 13.89) -> list[Path]:
    """Write node / edge / connection files; ``netconvert`` turns them into ``intersection.net.xml``."""
    CFG.mkdir(parents=True, exist_ok=True)
    L = arm_length_m
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="C", x="0", y="0", type="traffic_light", tl="TL")
    for nid, (x, y) in {"N": (0, L), "S": (0, -L), "E": (L, 0), "W": (-L, 0)}.items():
        ET.SubElement(nodes, "node", id=nid, x=str(x), y=str(y))
    edges = ET.Element("edges")
    for a in APPROACHES:
        ET.SubElement(edges, "edge", id=f"{a}2C", **{"from": a, "to": "C"}, numLanes="1", speed=str(speed_mps))
        ET.SubElement(edges, "edge", id=f"C2{a}", **{"from": "C", "to": a}, numLanes="1", speed=str(speed_mps))
    cons = ET.Element("connections")
    for a, b in OPPOSITE.items():           # straight-through only: no turning conflicts inside a phase
        ET.SubElement(cons, "connection", **{"from": f"{a}2C", "to": f"C2{b}"})
    paths = []
    for name, root in (("intersection.nod.xml", nodes), ("intersection.edg.xml", edges), ("intersection.con.xml", cons)):
        p = CFG / name
        ET.indent(root)
        ET.ElementTree(root).write(p, encoding="utf-8", xml_declaration=True)
        paths.append(p)
    return paths


def convert_network() -> Path:
    """Run ``netconvert`` (requires SUMO on PATH)."""
    if shutil.which("netconvert") is None:
        raise RuntimeError("netconvert not found: install SUMO and set SUMO_HOME / PATH (see simulation/README.md)")
    net = CFG / "intersection.net.xml"
    subprocess.run(["netconvert", "--node-files", str(CFG / "intersection.nod.xml"), "--edge-files", str(CFG / "intersection.edg.xml"),
                    "--connection-files", str(CFG / "intersection.con.xml"), "-o", str(net)], check=True)
    return net


# --------------------------------------------------------------------------- #
# Demand: seeded Poisson arrivals with a time-varying, deliberately asymmetric profile
# --------------------------------------------------------------------------- #
def demand_profile(duration_s: int = 3600) -> list[tuple[int, int, dict[str, float]]]:
    """(start, end, veh/h per approach): NS heavier in the first half, EW heavier in the second."""
    half = duration_s // 2
    return [(0, half, {"N": 500, "S": 500, "E": 200, "W": 200}), (half, duration_s, {"N": 200, "S": 200, "E": 500, "W": 500})]


def generate_routes(seed: int, duration_s: int = 3600, path: Path | None = None) -> tuple[Path, dict[str, int]]:
    """Write a route file with Poisson arrivals; returns (path, vehicles per approach)."""
    rng = random.Random(seed)
    vehicles = []
    for start, end, rates in demand_profile(duration_s):
        for a, per_hour in rates.items():
            t = start
            while True:
                t += rng.expovariate(per_hour / 3600.0)
                if t >= end:
                    break
                vehicles.append((t, a))
    vehicles.sort()
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", length="5", minGap="2.5", maxSpeed="13.89")
    for a, b in OPPOSITE.items():
        ET.SubElement(root, "route", id=f"r_{a}", edges=f"{a}2C C2{b}")
    for i, (t, a) in enumerate(vehicles):
        ET.SubElement(root, "vehicle", id=f"v{i}", type="car", route=f"r_{a}", depart=f"{t:.2f}")
    path = path or (OUT / f"routes_seed{seed}.rou.xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    counts = {a: sum(1 for _, x in vehicles if x == a) for a in APPROACHES}
    return path, counts


# --------------------------------------------------------------------------- #
# One TraCI episode (NOT executed during development: SUMO was unavailable)
# --------------------------------------------------------------------------- #
def run_episode(policy_name: str, seed: int, duration_s: int = 3600) -> dict[str, float]:
    import traci                                            # imported lazily so the module imports without SUMO

    net = CFG / "intersection.net.xml"
    routes, _ = generate_routes(seed, duration_s)
    tripinfo = OUT / f"tripinfo_{policy_name}_seed{seed}.xml"
    traci.start(["sumo", "-n", str(net), "-r", str(routes), "--seed", str(seed), "--no-step-log", "true",
                 "--time-to-teleport", "-1", "--tripinfo-output", str(tripinfo), "--no-warnings", "true"])
    try:
        policy = sp.FixedTimePolicy(30, 30) if policy_name == "fixed" else sp.QueueAwarePolicy(margin=2.0)
        controller = sp.SignalController(policy, sp.Timings())
        links = traci.trafficlight.getControlledLinks("TL")
        link_group = [GROUP[lk[0][0][0]] for lk in links]            # incoming lane id starts with the approach letter
        lanes = {g: [f"{a}2C_0" for a in APPROACHES if GROUP[a] == g] for g in (sp.NS, sp.EW)}
        queue_sum, max_queue = 0.0, 0.0
        for _ in range(duration_s):
            queues = {g: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes[g]) for g in lanes}
            colours = controller.step(queues)
            traci.trafficlight.setRedYellowGreenState("TL", "".join(colours[g] for g in link_group))
            traci.simulationStep()
            total = sum(queues.values()); queue_sum += total; max_queue = max(max_queue, total)
    finally:
        traci.close()
    trips = ET.parse(tripinfo).getroot().findall("tripinfo")
    wait = np.array([float(t.get("waitingTime")) for t in trips]); dur = np.array([float(t.get("duration")) for t in trips])
    return {"policy": policy_name, "seed": seed, "completed_trips": len(trips), "mean_waiting_time_s": float(wait.mean()),
            "mean_travel_time_s": float(dur.mean()), "mean_queue_vehicles": queue_sum / duration_s, "max_queue_vehicles": max_queue}
