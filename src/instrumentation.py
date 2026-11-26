"""Inactive instrumentation hooks for standalone research components."""

from typing import Any, Mapping


def record_episode_reset(
    *, soc0: float, dt_hours: float, e_bat_kwh: float, replacement_cost_y: float
) -> None:
    pass


def record_environment_step(info: Mapping[str, Any]) -> None:
    pass


def assert_environment_step_budget() -> None:
    pass


def record_hv_call() -> None:
    pass


def record_policy_update() -> None:
    pass
