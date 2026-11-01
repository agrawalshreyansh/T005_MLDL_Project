"""Traffic-signal policies for the *simulated* two-phase intersection (SUMO, optional).

This module is pure Python and has no SUMO dependency, so the safety logic can be unit-tested
anywhere (``tests/test_signal_policies.py``).  It never talks to real traffic infrastructure.

Safety by construction: a policy only answers "should the current green end now?".  The
:class:`SignalController` state machine enforces, independently of the policy,

* a minimum and a maximum green time,
* a yellow interval after every green,
* an all-red clearance interval before the conflicting phase turns green,
* and it can never show green/yellow to the north-south and east-west approaches at the same time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

NS, EW = "NS", "EW"
OTHER = {NS: EW, EW: NS}


@dataclass(frozen=True)
class Timings:
    """Signal timing limits in seconds."""

    min_green: int = 10
    max_green: int = 60
    yellow: int = 3
    all_red: int = 2

    def __post_init__(self) -> None:
        if not (0 < self.min_green <= self.max_green) or self.yellow < 1 or self.all_red < 1:
            raise ValueError("need 0 < min_green <= max_green, yellow >= 1, all_red >= 1")


class Policy(Protocol):
    """Decides whether the current green should end (the controller still enforces the limits)."""

    def should_switch(self, green: str, elapsed: int, queues: Mapping[str, float]) -> bool: ...


class FixedTimePolicy:
    """Pre-timed plan: each phase gets a fixed green time, ignoring the traffic."""

    def __init__(self, green_ns: int = 30, green_ew: int = 30) -> None:
        self.green = {NS: green_ns, EW: green_ew}

    def should_switch(self, green: str, elapsed: int, queues: Mapping[str, float]) -> bool:
        return elapsed >= self.green[green]


class ScheduledFixedTimePolicy:
    """Pre-timed plan whose green times change by time of day: ``schedule = [(end_s, green_ns, green_ew), ...]``.

    The controller tells the policy the simulation time through :meth:`on_step`.  This is how fixed-time plans are
    used in practice (different plans for different periods, designed from historical counts).
    """

    def __init__(self, schedule: list[tuple[int, int, int]]) -> None:
        if not schedule or any(g <= 0 for _, a, b in schedule for g in (a, b)):
            raise ValueError("schedule needs positive green times")
        self.schedule = sorted(schedule)
        self.clock = 0

    def on_step(self, clock: int) -> None:
        self.clock = clock

    def current(self) -> tuple[int, int]:
        for end, g_ns, g_ew in self.schedule:
            if self.clock < end:
                return g_ns, g_ew
        return self.schedule[-1][1], self.schedule[-1][2]

    def should_switch(self, green: str, elapsed: int, queues: Mapping[str, float]) -> bool:
        g_ns, g_ew = self.current()
        return elapsed >= (g_ns if green == NS else g_ew)


def webster_greens(flow_ns: float, flow_ew: float, timings: "Timings" = None, saturation_flow: float = 1800.0,
                   cycle_bounds: tuple[float, float] = (30.0, 120.0)) -> tuple[int, int, float]:
    """Webster's optimal cycle and green split for a two-phase signal.

    ``flow_*`` is the critical (largest) approach flow of each phase in veh/h, ``saturation_flow`` in veh/h/lane.
    Cycle ``C0 = (1.5 L + 5) / (1 - Y)`` with lost time ``L`` = (yellow + all-red) per phase and ``Y`` the sum of
    flow ratios ``y = q / s``; effective green ``C0 - L`` is split in proportion to ``y`` and each green is clipped
    to the controller's min/max green.  Returns ``(green_ns, green_ew, cycle)``.
    """
    timings = timings or Timings()
    y_ns, y_ew = flow_ns / saturation_flow, flow_ew / saturation_flow
    big_y = y_ns + y_ew
    if big_y >= 0.95:
        raise ValueError(f"demand too close to capacity for Webster's formula (Y = {big_y:.2f})")
    lost = 2 * (timings.yellow + timings.all_red)
    cycle = min(max((1.5 * lost + 5) / (1 - big_y), cycle_bounds[0]), cycle_bounds[1])
    effective = cycle - lost
    g_ns = int(round(min(max(effective * y_ns / big_y, timings.min_green), timings.max_green)))
    g_ew = int(round(min(max(effective * y_ew / big_y, timings.min_green), timings.max_green)))
    return g_ns, g_ew, float(cycle)


class QueueAwarePolicy:
    """Serve the longer queue: leave a green when the waiting approach is clearly more loaded.

    ``queues[phase]`` is the number of halted vehicles on the approaches of that phase (observed in the
    simulator).  The current green ends when its own queue is empty while the other has vehicles
    (gap-out), or when the other queue exceeds the current one by more than ``margin``.
    An optional per-phase ``forecast`` adds expected arrivals to the queues (a hook for a demand forecast).
    """

    def __init__(self, margin: float = 2.0, forecast_weight: float = 0.0) -> None:
        self.margin, self.forecast_weight = margin, forecast_weight
        self.forecast: dict[str, float] = {NS: 0.0, EW: 0.0}

    def pressure(self, phase: str, queues: Mapping[str, float]) -> float:
        return queues[phase] + self.forecast_weight * self.forecast.get(phase, 0.0)

    def should_switch(self, green: str, elapsed: int, queues: Mapping[str, float]) -> bool:
        mine, other = self.pressure(green, queues), self.pressure(OTHER[green], queues)
        return (mine == 0 and other > 0) or (other > mine + self.margin)


class SignalController:
    """State machine ``NS green -> NS yellow -> all red -> EW green -> EW yellow -> all red -> ...``."""

    def __init__(self, policy: Policy, timings: Timings = Timings(), start: str = NS) -> None:
        self.policy, self.t = policy, timings
        self.phase, self.stage, self.elapsed = start, "green", 0      # stage in {"green", "yellow", "all_red"}
        self.clock = 0                                                # simulation seconds since the start

    def step(self, queues: Mapping[str, float]) -> dict[str, str]:
        """Show the current state for one second, then advance; returns the colours shown ('G', 'y', 'r')."""
        if hasattr(self.policy, "on_step"):
            self.policy.on_step(self.clock)
        self.clock += 1
        shown = self.colours()
        self.elapsed += 1                                   # seconds this stage has now been displayed
        if self.stage == "green":
            ended = self.elapsed >= self.t.max_green or (
                self.elapsed >= self.t.min_green and self.policy.should_switch(self.phase, self.elapsed, queues))
            if ended:
                self.stage, self.elapsed = "yellow", 0
        elif self.stage == "yellow":
            if self.elapsed >= self.t.yellow:
                self.stage, self.elapsed = "all_red", 0
        else:  # all_red
            if self.elapsed >= self.t.all_red:
                self.phase, self.stage, self.elapsed = OTHER[self.phase], "green", 0
        return shown

    def colours(self) -> dict[str, str]:
        state = {NS: "r", EW: "r"}
        if self.stage == "green":
            state[self.phase] = "G"
        elif self.stage == "yellow":
            state[self.phase] = "y"
        return state


def run_trace(policy: Policy, queue_fn, seconds: int, timings: Timings = Timings()) -> list[dict[str, str]]:
    """Drive a controller for ``seconds`` with ``queue_fn(t) -> {NS: q, EW: q}``; returns the colour trace."""
    ctrl = SignalController(policy, timings)
    return [ctrl.step(queue_fn(t)) for t in range(seconds)]


def check_trace_is_safe(trace: list[dict[str, str]], timings: Timings = Timings()) -> None:
    """Raise AssertionError if the trace ever violates a safety constraint."""
    for t, s in enumerate(trace):
        assert not (s[NS] in "Gy" and s[EW] in "Gy"), f"conflicting greens at t={t}"
    for phase in (NS, EW):
        colours = [s[phase] for s in trace]
        runs, start = [], 0
        for i in range(1, len(colours) + 1):
            if i == len(colours) or colours[i] != colours[start]:
                runs.append((colours[start], start, i - start))
                start = i
        for k, (col, begin, length) in enumerate(runs):
            complete = k < len(runs) - 1                       # the last run may be cut off by the end of the trace
            if col == "G" and complete:
                assert timings.min_green <= length <= timings.max_green, f"{phase}: green of {length}s at t={begin}"
            if col == "y" and complete:
                assert length == timings.yellow, f"{phase}: yellow of {length}s at t={begin}"
            if col == "G" and k + 1 < len(runs):
                assert runs[k + 1][0] == "y", f"{phase}: green not followed by yellow at t={begin + length}"
