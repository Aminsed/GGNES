"""ValidationError and error handling."""


class ValidationError(Exception):
    """Structured validation error for robust handling."""

    def __init__(self, error_type: str, message: str, **details):
        self.error_type = error_type
        self.message = message
        self.details = details

    def __str__(self):
        return f"[{self.error_type}] {self.message}"


class NodeError(ValidationError):
    """Node-specific validation error."""

    def __init__(self, node_id: int, error_type: str, message: str, **details):
        super().__init__(error_type, message, node_id=node_id, **details)
        self.node_id = node_id


class EdgeError(ValidationError):
    """Edge-specific validation error."""

    def __init__(self, edge_id, error_type: str, message: str, **details):
        super().__init__(error_type, message, edge_id=edge_id, **details)
        self.edge_id = edge_id
