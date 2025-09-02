"""
Test module for scaffold and project structure.
Tests import smoke tests for all submodules per M1 milestone.
"""


def test_scaffold_import_smoke():
    """[T-scaffold-01] Import smoke tests for each submodule."""
    # Test main module import
    import ggnes  # noqa: F401

    # Test core submodules
    from ggnes.core import graph  # noqa: F401
    from ggnes.core import node  # noqa: F401
    from ggnes.core import edge  # noqa: F401
    from ggnes.core import primitives  # noqa: F401
    from ggnes.core import id_manager  # noqa: F401

    # Test rules submodules
    from ggnes.rules import rule  # noqa: F401
    from ggnes.rules import conditions  # noqa: F401
    from ggnes.rules import predicates  # noqa: F401

    # Test generation submodules
    from ggnes.generation import network_gen  # noqa: F401
    from ggnes.generation import rule_engine  # noqa: F401
    from ggnes.generation import matching  # noqa: F401
    from ggnes.generation import oscillation  # noqa: F401

    # Test translation submodules
    from ggnes.translation import pytorch  # noqa: F401
    from ggnes.translation import state_manager  # noqa: F401

    # Test evolution submodules
    from ggnes.evolution import genotype  # noqa: F401
    from ggnes.evolution import operators  # noqa: F401
    from ggnes.evolution import selection  # noqa: F401

    # Test repair submodule
    from ggnes.repair import repair  # noqa: F401

    # Test utils submodules
    from ggnes.utils import rng_manager  # noqa: F401
    from ggnes.utils import serialization  # noqa: F401
    from ggnes.utils import validation  # noqa: F401

    # Test that we can import from main module
    from ggnes.core import Graph, Node, Edge  # noqa: F401
    from ggnes.generation import generate_network  # noqa: F401
    from ggnes.translation import to_pytorch_model  # noqa: F401
    from ggnes.repair import repair as repair_func  # noqa: F401

    # If we reach here, all imports succeeded
    assert True
