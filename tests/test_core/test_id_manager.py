"""
Comprehensive tests for IDManager to ensure strict adherence to project_guide.md.
"""

import uuid

from ggnes.core import Edge, Graph, Node
from ggnes.core.graph import IDStrategy, NodeType
from ggnes.core.id_manager import IDManager


class TestIDManagerBasics:
    """Basic tests for IDManager initialization and strategies."""

    def test_id_manager_initialization_with_strategy(self):
        """Test IDManager initializes with provided strategy."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)
        assert manager.id_strategy == IDStrategy.HYBRID

        manager2 = IDManager(id_strategy=IDStrategy.LOCAL_ONLY)
        assert manager2.id_strategy == IDStrategy.LOCAL_ONLY

        manager3 = IDManager(id_strategy=IDStrategy.GLOBAL_ONLY)
        assert manager3.id_strategy == IDStrategy.GLOBAL_ONLY

    def test_id_manager_default_strategy(self):
        """Test IDManager defaults to HYBRID strategy."""
        manager = IDManager()
        assert manager.id_strategy == IDStrategy.HYBRID

    def test_id_manager_mapping_structures(self):
        """Test IDManager initializes mapping structures."""
        manager = IDManager()

        assert isinstance(manager.global_to_local, dict)
        assert isinstance(manager.local_to_global, dict)
        assert len(manager.global_to_local) == 0
        assert len(manager.local_to_global) == 0


class TestIDManagerNodeRegistration:
    """Tests for node registration across different strategies."""

    def test_register_node_local_only_no_mapping(self):
        """[T-id-01] LOCAL_ONLY: no mapping occurs."""
        manager = IDManager(id_strategy=IDStrategy.LOCAL_ONLY)

        # Create a node
        node = Node(
            node_id=5,
            node_type=NodeType.HIDDEN,
            activation_function="relu",
            attributes={"output_size": 10}
        )

        # Register it
        manager.register_node(node, context_id="test_context")

        # No mappings should be created
        assert len(manager.global_to_local) == 0
        assert len(manager.local_to_global) == 0

        # Node should not get a global_id
        assert node.global_id is None

    def test_register_node_hybrid_creates_mappings(self):
        """[T-id-01] HYBRID: nodes get global UUID and mappings are registered."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Create a node without global_id
        node = Node(
            node_id=10,
            node_type=NodeType.INPUT,
            activation_function="linear",
            attributes={"output_size": 20}
        )

        # Initially no global_id
        assert node.global_id is None

        # Register it
        context_id = "hybrid_context"
        manager.register_node(node, context_id)

        # Should now have global_id
        assert node.global_id is not None
        assert isinstance(node.global_id, uuid.UUID)

        # Check mappings
        assert context_id in manager.global_to_local
        assert context_id in manager.local_to_global

        # Check global -> local mapping
        assert node.global_id in manager.global_to_local[context_id]
        assert manager.global_to_local[context_id][node.global_id] == 10

        # Check local -> global mapping
        assert 'node' in manager.local_to_global[context_id]
        assert 10 in manager.local_to_global[context_id]['node']
        assert manager.local_to_global[context_id]['node'][10] == node.global_id

    def test_register_node_global_only_creates_mappings(self):
        """[T-id-01] GLOBAL_ONLY: identity mapping for nodes."""
        manager = IDManager(id_strategy=IDStrategy.GLOBAL_ONLY)

        # Create a node
        node = Node(
            node_id=15,
            node_type=NodeType.OUTPUT,
            activation_function="sigmoid",
            attributes={"output_size": 5}
        )

        context_id = "global_context"
        manager.register_node(node, context_id)

        # Should have global_id
        assert node.global_id is not None

        # Should have mappings
        assert manager.global_to_local[context_id][node.global_id] == 15
        assert manager.local_to_global[context_id]['node'][15] == node.global_id

    def test_register_node_preserves_existing_global_id(self):
        """Test that existing global_id is preserved during registration."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Create node with existing global_id
        existing_uuid = uuid.uuid4()
        node = Node(
            node_id=20,
            node_type=NodeType.HIDDEN,
            activation_function="tanh",
            attributes={"output_size": 30},
            global_id=existing_uuid
        )

        manager.register_node(node, "test_context")

        # Should keep the same global_id
        assert node.global_id == existing_uuid

        # Should be in mappings
        assert manager.global_to_local["test_context"][existing_uuid] == 20


class TestIDManagerEdgeRegistration:
    """Tests for edge registration across different strategies."""

    def test_register_edge_local_only_no_mapping(self):
        """[T-id-01] LOCAL_ONLY: no edge mapping occurs."""
        manager = IDManager(id_strategy=IDStrategy.LOCAL_ONLY)

        # Create an edge
        edge = Edge(
            edge_id=100,
            source_node_id=1,
            target_node_id=2,
            weight=0.5
        )

        manager.register_edge(edge, "test_context")

        # No mappings should exist
        assert len(manager.global_to_local) == 0
        assert len(manager.local_to_global) == 0

    def test_register_edge_hybrid_local_edge_id(self):
        """[T-id-01] HYBRID: edges get global UUID and local_edge_id."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Create edge with UUID (as Graph would in HYBRID mode)
        edge_uuid = uuid.uuid4()
        edge = Edge(
            edge_id=edge_uuid,
            source_node_id=1,
            target_node_id=2,
            weight=0.5
        )
        edge.local_edge_id = 42  # Graph sets this in HYBRID mode

        context_id = "hybrid_edge_context"
        manager.register_edge(edge, context_id)

        # Check mappings exist
        assert context_id in manager.global_to_local
        assert context_id in manager.local_to_global

        # Check global -> local mapping
        assert edge_uuid in manager.global_to_local[context_id]
        assert manager.global_to_local[context_id][edge_uuid] == 42

        # Check local -> global mapping
        assert 'edge' in manager.local_to_global[context_id]
        assert 42 in manager.local_to_global[context_id]['edge']
        assert manager.local_to_global[context_id]['edge'][42] == edge_uuid

    def test_register_edge_global_only_uuid_mapping(self):
        """[T-id-01] GLOBAL_ONLY: edge UUID mapping."""
        manager = IDManager(id_strategy=IDStrategy.GLOBAL_ONLY)

        # Create edge with UUID
        edge_uuid = uuid.uuid4()
        edge = Edge(
            edge_id=edge_uuid,
            source_node_id=3,
            target_node_id=4,
            weight=0.7
        )

        context_id = "global_edge_context"
        manager.register_edge(edge, context_id)

        # For GLOBAL_ONLY, still track UUID (identity mapping)
        assert context_id in manager.global_to_local
        assert edge_uuid in manager.global_to_local[context_id]

        # The "local" ID is just the UUID itself for GLOBAL_ONLY
        assert manager.global_to_local[context_id][edge_uuid] == edge_uuid


class TestIDManagerLookupMethods:
    """Tests for ID lookup methods."""

    def test_get_local_node_id(self):
        """Test get_local_node_id method."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Register a node
        node = Node(
            node_id=25,
            node_type=NodeType.HIDDEN,
            activation_function="elu",
            attributes={"output_size": 15}
        )

        context_id = "lookup_context"
        manager.register_node(node, context_id)

        # Lookup by global_id
        local_id = manager.get_local_node_id(node.global_id, context_id)
        assert local_id == 25

        # Non-existent context
        assert manager.get_local_node_id(node.global_id, "wrong_context") is None

        # Non-existent global_id
        assert manager.get_local_node_id(uuid.uuid4(), context_id) is None

    def test_get_global_node_id(self):
        """Test get_global_node_id method."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Register a node
        node = Node(
            node_id=30,
            node_type=NodeType.INPUT,
            activation_function="linear",
            attributes={"output_size": 10}
        )

        context_id = "global_lookup"
        manager.register_node(node, context_id)

        # Lookup by local_id
        global_id = manager.get_global_node_id(30, context_id)
        assert global_id == node.global_id

        # Non-existent lookups
        assert manager.get_global_node_id(30, "wrong_context") is None
        assert manager.get_global_node_id(999, context_id) is None

    def test_get_local_edge_id(self):
        """Test get_local_edge_id method."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Create and register edge
        edge_uuid = uuid.uuid4()
        edge = Edge(
            edge_id=edge_uuid,
            source_node_id=1,
            target_node_id=2,
            weight=0.5
        )
        edge.local_edge_id = 55

        context_id = "edge_lookup"
        manager.register_edge(edge, context_id)

        # Lookup
        local_id = manager.get_local_edge_id(edge_uuid, context_id)
        assert local_id == 55

        # Non-existent lookups
        assert manager.get_local_edge_id(edge_uuid, "wrong_context") is None
        assert manager.get_local_edge_id(uuid.uuid4(), context_id) is None

    def test_get_global_edge_id(self):
        """Test get_global_edge_id method."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Create and register edge
        edge_uuid = uuid.uuid4()
        edge = Edge(
            edge_id=edge_uuid,
            source_node_id=3,
            target_node_id=4,
            weight=0.3
        )
        edge.local_edge_id = 77

        context_id = "edge_global_lookup"
        manager.register_edge(edge, context_id)

        # Lookup
        global_id = manager.get_global_edge_id(77, context_id)
        assert global_id == edge_uuid

        # Non-existent lookups
        assert manager.get_global_edge_id(77, "wrong_context") is None
        assert manager.get_global_edge_id(999, context_id) is None


class TestIDManagerIntegration:
    """Integration tests with Graph class."""

    def test_id_manager_with_graph_local_only(self):
        """Test IDManager with Graph using LOCAL_ONLY strategy."""
        graph = Graph(config={'id_strategy': 'LOCAL_ONLY'})
        manager = IDManager(id_strategy=IDStrategy.LOCAL_ONLY)

        # Add nodes
        n1 = graph.add_node({
            "node_type": NodeType.INPUT,
            "activation_function": "linear",
            "attributes": {"output_size": 10}
        })
        n2 = graph.add_node({
            "node_type": NodeType.OUTPUT,
            "activation_function": "sigmoid",
            "attributes": {"output_size": 5}
        })

        # Add edge
        edge_id = graph.add_edge(n1, n2, {"weight": 0.5})

        # Register with IDManager
        context = "local_only_test"
        manager.register_node(graph.nodes[n1], context)
        manager.register_node(graph.nodes[n2], context)

        edge = graph.find_edge_by_id(edge_id)
        manager.register_edge(edge, context)

        # No mappings should exist
        assert len(manager.global_to_local) == 0
        assert len(manager.local_to_global) == 0

    def test_id_manager_with_graph_hybrid(self):
        """Test IDManager with Graph using HYBRID strategy."""
        graph = Graph(config={'id_strategy': 'HYBRID'})
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Add nodes
        n1 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "relu",
            "attributes": {"output_size": 20}
        })

        # Node should have global_id
        node1 = graph.nodes[n1]
        assert node1.global_id is not None

        # Add edge
        n2 = graph.add_node({
            "node_type": NodeType.HIDDEN,
            "activation_function": "tanh",
            "attributes": {"output_size": 15}
        })
        edge_id = graph.add_edge(n1, n2, {"weight": 0.7})

        # Edge should have UUID and local_edge_id
        edge = graph.find_edge_by_id(edge_id)
        assert isinstance(edge.edge_id, uuid.UUID)
        assert isinstance(edge.local_edge_id, int)

        # Register everything
        context = "hybrid_test"
        manager.register_node(graph.nodes[n1], context)
        manager.register_node(graph.nodes[n2], context)
        manager.register_edge(edge, context)

        # Verify mappings
        assert manager.get_local_node_id(node1.global_id, context) == n1
        assert manager.get_global_node_id(n1, context) == node1.global_id
        assert manager.get_local_edge_id(edge_id, context) == edge.local_edge_id
        assert manager.get_global_edge_id(edge.local_edge_id, context) == edge_id

    def test_id_manager_multiple_contexts(self):
        """Test IDManager handles multiple contexts independently."""
        manager = IDManager(id_strategy=IDStrategy.HYBRID)

        # Create nodes
        node1 = Node(
            node_id=1,
            node_type=NodeType.INPUT,
            activation_function="linear",
            attributes={"output_size": 10}
        )
        node2 = Node(
            node_id=2,
            node_type=NodeType.OUTPUT,
            activation_function="softmax",
            attributes={"output_size": 5}
        )

        # Register in different contexts
        manager.register_node(node1, "context_A")
        manager.register_node(node2, "context_B")

        # Each context should be independent
        assert "context_A" in manager.global_to_local
        assert "context_B" in manager.global_to_local

        # node1 only in context_A
        assert manager.get_local_node_id(node1.global_id, "context_A") == 1
        assert manager.get_local_node_id(node1.global_id, "context_B") is None

        # node2 only in context_B
        assert manager.get_local_node_id(node2.global_id, "context_B") == 2
        assert manager.get_local_node_id(node2.global_id, "context_A") is None
