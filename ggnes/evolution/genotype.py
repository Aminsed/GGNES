"""Genotype class for evolutionary engine (project_guide.md §9.1)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Genotype:
    """Genetic representation as ordered list of rules.

    Attributes:
        genotype_id: Unique identifier for this genotype
        rules: Ordered list of rules forming the grammar
        fitness: Optional fitness value assigned by evaluation
        repair_impact: Aggregate repair impact for this genotype
        history: Additional bookkeeping data
    """

    genotype_id: uuid.UUID = field(default_factory=uuid.uuid4)
    rules: list[Any] = field(default_factory=list)
    fitness: float | None = None
    repair_impact: float = 0.0
    history: dict = field(default_factory=dict)


__all__ = ["Genotype"]
