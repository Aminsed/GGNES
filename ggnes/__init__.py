"""
GGNES - Graph Grammar Neuroevolution System

A system for evolving neural network architectures by evolving the generative
graph rewriting rules (grammar) that construct them.
"""

__version__ = "0.1.0"

# Expose common submodules for convenience
from .core import *  # noqa: F401,F403
from .generation import *  # noqa: F401,F403
from .repair import *  # noqa: F401,F403
from .rules import *  # noqa: F401,F403
from .translation import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

# Expose configuration presets as top-level names (used in guide examples)
try:  # pragma: no cover - import guarded for packaging safety
    from .config import PRESET_MINIMAL, PRESET_RESEARCH, PRESET_STANDARD  # noqa: F401
except Exception:  # pragma: no cover
    PRESET_MINIMAL = PRESET_STANDARD = PRESET_RESEARCH = None  # type: ignore
