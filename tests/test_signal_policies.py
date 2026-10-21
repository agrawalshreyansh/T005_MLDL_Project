"""Safety logic of the simulated signal controller (no SUMO needed)."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
import signal_policies as sp  # noqa: E402


def test_fixed_time_cycle_respects_all_constraints_and_never_conflicts():
    trace = sp.run_trace(sp.FixedTimePolicy(30, 20), lambda t: {sp.NS: 0, sp.EW: 0}, 600)
    sp.check_trace_is_safe(trace)
    ns_green = sum(s[sp.NS] == "G" for s in trace)
    ew_green = sum(s[sp.EW] == "G" for s in trace)
    assert ns_green > ew_green > 0                                  # 30 s vs 20 s plans


def test_queue_aware_policy_is_safe_under_random_queues():
    rng = random.Random(0)
    for _ in range(20):
        qs = {t: {sp.NS: rng.randint(0, 25), sp.EW: rng.randint(0, 25)} for t in range(900)}
        trace = sp.run_trace(sp.QueueAwarePolicy(), lambda t: qs[t], 900)
        sp.check_trace_is_safe(trace)


def test_queue_aware_gives_green_to_the_longer_queue_but_respects_min_green():
    timings = sp.Timings(min_green=10, max_green=60, yellow=3, all_red=2)
    trace = sp.run_trace(sp.QueueAwarePolicy(margin=2), lambda t: {sp.NS: 0, sp.EW: 20}, 40, timings)
    ns_runs = _runs([s[sp.NS] for s in trace])
    assert ns_runs[0][0] == "G" and len(ns_runs[0]) == timings.min_green     # first green lasts exactly min_green (policy wants out at once)
    assert len(ns_runs[1]) == timings.yellow and ns_runs[1][0] == "y"        # then yellow
    first_ew_green = next(i for i, s in enumerate(trace) if s[sp.EW] == "G")
    assert first_ew_green == timings.min_green + timings.yellow + timings.all_red    # loaded approach served right after clearance
    assert all(s[sp.EW] == "G" for s in trace[first_ew_green:])                      # and keeps the green: its queue is the longer one


def test_clearance_times_between_conflicting_greens_are_exact():
    timings = sp.Timings(min_green=5, max_green=15, yellow=3, all_red=2)
    trace = sp.run_trace(sp.FixedTimePolicy(10, 10), lambda t: {sp.NS: 0, sp.EW: 0}, 120, timings)
    sp.check_trace_is_safe(trace, timings)
    ns = [s[sp.NS] for s in trace]; ew = [s[sp.EW] for s in trace]
    last_ns_g = max(i for i, c in enumerate(ns[:30]) if c == "G"); first_ew_g = min(i for i, c in enumerate(ew) if c == "G")
    assert first_ew_g - last_ns_g - 1 == timings.yellow + timings.all_red      # 3 s yellow + 2 s all-red between the greens


def test_max_green_is_enforced_even_if_the_policy_never_wants_to_switch():
    class Never:
        def should_switch(self, green, elapsed, queues):
            return False
    timings = sp.Timings(min_green=5, max_green=20, yellow=3, all_red=2)
    trace = sp.run_trace(Never(), lambda t: {sp.NS: 50, sp.EW: 50}, 300, timings)
    sp.check_trace_is_safe(trace, timings)
    longest = max(len(list(g)) for g in _runs([s[sp.NS] for s in trace]) if g and g[0] == "G")
    assert longest <= 20


def _runs(seq):
    out, cur = [], []
    for x in seq:
        if cur and x != cur[-1]:
            out.append(cur); cur = []
        cur.append(x)
    return out + [cur]


def test_invalid_timings_are_rejected():
    with pytest.raises(ValueError):
        sp.Timings(min_green=30, max_green=10)
    with pytest.raises(ValueError):
        sp.Timings(yellow=0)


def test_safety_checker_detects_a_conflict():
    bad = [{sp.NS: "G", sp.EW: "G"}]
    with pytest.raises(AssertionError):
        sp.check_trace_is_safe(bad)
