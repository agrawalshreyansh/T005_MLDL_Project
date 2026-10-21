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

    def step(self, queues: Mapping[str, float]) -> dict[str, str]:
        """Show the current state for one second, then advance; returns the colours shown ('G', 'y', 'r')."""
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
