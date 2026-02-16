"""Geometry contract and helpers for Rocket League field calculations."""

from __future__ import annotations

from constants import FIELD_HALF_Y, GOAL_HALF_WIDTH, GOAL_WIDTH


def assert_geometry_contract() -> None:
    """Validate invariants for shared field/goal geometry constants."""
    if GOAL_WIDTH != 2 * GOAL_HALF_WIDTH:
        raise ValueError(
            f"Invalid goal geometry: GOAL_WIDTH={GOAL_WIDTH} must equal "
            f"2 * GOAL_HALF_WIDTH={2 * GOAL_HALF_WIDTH}."
        )


def goal_target_y(team: str) -> float:
    """Return defending-goal Y coordinate for the given shooting team."""
    return float(FIELD_HALF_Y if team == "Blue" else -FIELD_HALF_Y)


def goal_mouth_x_bounds() -> tuple[float, float]:
    """Return (left_x, right_x) bounds of the goal mouth on the X axis."""
    return float(-GOAL_HALF_WIDTH), float(GOAL_HALF_WIDTH)


assert_geometry_contract()
