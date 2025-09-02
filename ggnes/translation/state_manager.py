"""StateManager for recurrent state handling.

Implements the API specified in project_guide.md §8.2 State Management.
"""

from __future__ import annotations

from typing import Any


class StateManager:
    """Manages runtime state for recurrent nodes.

    Args:
        max_sequence_length: Optional maximum sequence length for memory management.
                             If provided, warnings are issued for longer sequences.
    """

    def __init__(self, max_sequence_length: int | None = None):
        self.states: dict[int, Any] = {}
        self.prev_outputs: dict[int, Any] = {}
        self.max_sequence_length = max_sequence_length
        self.sequence_length = 0
        self.batch_size: int | None = None
        self.device: Any = None

    def initialize(self, batch_size: int, device: Any) -> None:
        """Initialize for a forward pass.

        Args:
            batch_size: Batch size
            device: Backend device handle
        """
        self.batch_size = batch_size
        self.device = device

    def get_state(self, node_id: int, hidden_size: int, state_type: str = "default"):
        """Get or create state for a node.

        Args:
            node_id: Node ID
            hidden_size: Hidden size
            state_type: Type of state (default/lstm)

        Returns:
            State tensor(s)
        """
        # Lazy import to avoid hard torch dependency for non-PyTorch code paths
        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover - exercised in torch-available CI
            torch = None  # type: ignore

        if node_id not in self.states or self.states[node_id] is None:
            if state_type == "lstm":
                if torch is None:
                    # Minimal fallback: tuples of zeros lists
                    h = [0.0] * hidden_size
                    c = [0.0] * hidden_size
                else:  # pragma: no cover - executed only when torch is installed
                    h = torch.zeros(self.batch_size, hidden_size, device=self.device)
                    c = torch.zeros(self.batch_size, hidden_size, device=self.device)
                self.states[node_id] = (h, c)
            else:
                if torch is None:
                    state = [0.0] * hidden_size
                else:  # pragma: no cover - executed only when torch is installed
                    state = torch.zeros(self.batch_size, hidden_size, device=self.device)
                self.states[node_id] = state

        return self.states[node_id]

    def update_state(self, node_id: int, new_state) -> None:
        """Update state for a node."""
        self.states[node_id] = new_state

    def store_output(self, node_id: int, output) -> None:
        """Store output for use in next timestep."""
        self.prev_outputs[node_id] = output

        # Sequence length tracking (warning only)
        if self.max_sequence_length and self.sequence_length == 0:
            self.sequence_length = 1
        elif self.max_sequence_length and node_id in self.prev_outputs:
            self.sequence_length += 1
            if self.sequence_length > self.max_sequence_length:
                import warnings

                msg = (
                    f"Sequence length {self.sequence_length} exceeds configured maximum "
                    f"{self.max_sequence_length}. Consider resetting states or increasing limit."
                )
                warnings.warn(msg, ResourceWarning)

    def get_prev_output(self, node_id: int):
        """Get previous timestep output or None."""
        return self.prev_outputs.get(node_id)

    def reset(self) -> None:
        """Reset all states and sequence tracking."""
        self.states.clear()
        self.prev_outputs.clear()
        self.sequence_length = 0
