"""Failure floor policy (project_guide.md §12.3).

Applies a relaxation policy to the configuration when too many genotypes
fail in a generation. This function mutates the provided config dict,
mirroring the behavior described in the guide.
"""

from __future__ import annotations

from typing import Any


def apply_failure_floor_policy(generation_results: list[dict[str, Any]] | None, config: dict) -> None:
    """Apply failure floor policy when too many genotypes fail.

    Args:
        generation_results: List of result dicts containing a 'fitness' key.
        config: Configuration dict to be updated in-place.
    """
    if not generation_results:
        return

    total = len(generation_results)
    if total == 0:  # pragma: no cover - redundant due to falsy check above
        return

    failures = sum(1 for r in generation_results if r.get('fitness') == float('-inf'))
    failure_rate = failures / total

    threshold = float(config.get('failure_floor_threshold', 0.5))
    if failure_rate > threshold:
        # Relax constraints as per project guide
        max_iter = int(config.get('max_iterations', 50))
        # Reduce by 20% (floor to int)
        config['max_iterations'] = max(1, int(max_iter * 0.8))
        config['repair_strategy'] = 'AGGRESSIVE'

        # Note: Logging left to caller environment; keep function pure-ish aside from config mutation


__all__ = [
    'apply_failure_floor_policy',
]


