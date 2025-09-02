"""
M8 Oscillation detection tests per project_guide.md.

Covers:
- [T-osc-01] SIMPLE_REVERSAL within window and non-reversal cases
- [T-osc-02] STATE_FINGERPRINTING excludes recent N, detects cycles, handles no match
- [T-osc-03] GROWTH_MONITORING detects stagnation only when rules applied; otherwise false
- [T-osc-04] Aggregator precedence and per-strategy selection variants
"""

from __future__ import annotations

from ggnes.generation.oscillation import (
    detect_growth_monitoring,
    detect_oscillation,
    detect_simple_reversal,
    detect_state_fingerprinting,
)


class TestSimpleReversal:

    def test_reversal_detected_within_window(self):
        class R:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        history = [
            {"rule_info": ("DeleteNode", None)},
            {"rule_info": ("X", None)},
        ]
        cfg = {"rule_reverse_pairs": {"AddNode": "DeleteNode"}, "oscillation_history_window": 2}
        ok, reason = detect_simple_reversal(R("AddNode"), history, cfg)
        assert ok and "reverses previous" in reason

    def test_reversal_not_detected_when_no_pairs(self):
        class R:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        history = [{"rule_info": ("B", None)}]
        cfg = {"rule_reverse_pairs": {}, "oscillation_history_window": 5}
        ok, reason = detect_simple_reversal(R("A"), history, cfg)
        assert ok is False and reason is None

    def test_reversal_not_detected_outside_window(self):
        class R:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        history = [
            {"rule_info": ("DeleteNode", None)},
            {"rule_info": ("Other", None)},
            {"rule_info": ("Other2", None)},
        ]
        cfg = {"rule_reverse_pairs": {"AddNode": "DeleteNode"}, "oscillation_history_window": 1}
        ok, reason = detect_simple_reversal(R("AddNode"), history, cfg)
        assert ok is False and reason is None


class TestStateFingerprinting:
    def test_detects_cycle_excluding_recent(self):
        class G:
            def __init__(self, fp):
                self._fp = fp

            def compute_fingerprint(self):
                return self._fp

        class GH:
            def __init__(self, fps):
                self.fingerprints = fps

        gh = GH(["a", "b", "c", "a"])  # earlier "a" exists outside exclusion window
        g = G("a")
        ok, reason = detect_state_fingerprinting(g, gh, {"fingerprint_exclusion_window": 1})
        assert ok and "State cycle detected" in reason

    def test_no_cycle_when_recent_only_or_absent(self):
        class G:
            def __init__(self, fp):
                self._fp = fp

            def compute_fingerprint(self):
                return self._fp

        class GH:
            def __init__(self, fps):
                self.fingerprints = fps

        # Exclusion covers the only match
        gh1 = GH(["x", "y", "z"])  # no match
        g1 = G("q")
        ok1, reason1 = detect_state_fingerprinting(g1, gh1, {"fingerprint_exclusion_window": 1})
        assert ok1 is False and reason1 is None

        gh2 = GH(["p", "q"])  # match exists but exclusion window includes it
        g2 = G("q")
        ok2, reason2 = detect_state_fingerprinting(g2, gh2, {"fingerprint_exclusion_window": 1})
        assert ok2 is False and reason2 is None

    def test_zero_exclusion_window(self):
        class G:
            def compute_fingerprint(self):
                return "m"

        class GH:
            def __init__(self):
                self.fingerprints = ["m", "n"]

        ok, reason = detect_state_fingerprinting(
            G(), GH(), {"fingerprint_exclusion_window": 0}
        )
        assert ok and "State cycle detected" in reason


class TestGrowthMonitoring:
    def test_flags_stagnation_with_rule_applications(self):
        metrics = [
            {"num_nodes": 2, "num_edges": 1, "rule_applied": 1}
            for _ in range(5)
        ]
        ok, reason = detect_growth_monitoring(metrics, {"oscillation_check_depth": 5})
        assert ok and "No structural growth" in reason

    def test_no_flag_when_insufficient_depth(self):
        metrics = [
            {"num_nodes": 2, "num_edges": 1, "rule_applied": 1}
            for _ in range(3)
        ]
        ok, reason = detect_growth_monitoring(metrics, {"oscillation_check_depth": 5})
        assert ok is False and reason is None

    def test_no_flag_when_growth_changes_or_no_rules(self):
        metrics1 = [
            {"num_nodes": 2, "num_edges": 1, "rule_applied": 1},
            {"num_nodes": 3, "num_edges": 2, "rule_applied": 1},
            {"num_nodes": 3, "num_edges": 2, "rule_applied": 1},
            {"num_nodes": 3, "num_edges": 3, "rule_applied": 1},
            {"num_nodes": 3, "num_edges": 3, "rule_applied": 1},
        ]
        ok1, reason1 = detect_growth_monitoring(metrics1, {"oscillation_check_depth": 5})
        assert ok1 is False and reason1 is None

        metrics2 = [
            {"num_nodes": 2, "num_edges": 1, "rule_applied": 0}
            for _ in range(5)
        ]
        ok2, reason2 = detect_growth_monitoring(metrics2, {"oscillation_check_depth": 5})
        assert ok2 is False and reason2 is None


class TestAggregator:
    def test_all_strategies_sequence_and_unknown_strategy(self):
        class G:
            def compute_fingerprint(self):
                return "fp"

        class GH:
            def __init__(self):
                self.fingerprints = []
                self.action_history = []
                self.metrics = []

        class R:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        cfg = {"oscillation_strategy": ["STATE_FINGERPRINTING", "GROWTH_MONITORING"]}
        ok, reason = detect_oscillation(GH(), G(), R("A"), {}, cfg)
        assert ok is False and reason is None

        # Provide malformed type (string) for non-list and ensure normalization
        cfg2 = {"oscillation_strategy": "GROWTH_MONITORING"}
        ok2, reason2 = detect_oscillation(GH(), G(), R("A"), {}, cfg2)
        assert ok2 is False and reason2 is None

    def test_aggregator_state_success(self):
        class G:
            def __init__(self, fp):
                self._fp = fp

            def compute_fingerprint(self):
                return self._fp

        class GH:
            def __init__(self, fps):
                self.fingerprints = fps
                self.action_history = []
                self.metrics = []

        cfg = {"oscillation_strategy": ["STATE_FINGERPRINTING"], "fingerprint_exclusion_window": 1}
        gh = GH(["k", "m", "k"])  # prior 'k' exists
        ok, reason = detect_oscillation(gh, G("k"), object(), {}, cfg)
        assert ok and reason.startswith("[STATE_FINGERPRINTING]")

    def test_aggregator_growth_success_and_unknown_first(self):
        class G:
            def compute_fingerprint(self):
                return "nope"

        class GH:
            def __init__(self):
                self.fingerprints = []
                self.action_history = []
                self.metrics = [
                    {"num_nodes": 4, "num_edges": 5, "rule_applied": 1}
                    for _ in range(5)
                ]

        # Unknown strategy should be ignored; GROWTH_MONITORING should trigger
        cfg = {
            "oscillation_strategy": ["UNKNOWN", "GROWTH_MONITORING"],
            "oscillation_check_depth": 5,
        }
        ok, reason = detect_oscillation(GH(), G(), object(), {}, cfg)
        assert ok and reason.startswith("[GROWTH_MONITORING]")

    def test_aggregator_simple_false_then_state_true(self):
        class G:
            def __init__(self, fp):
                self._fp = fp

            def compute_fingerprint(self):
                return self._fp

        class GH:
            def __init__(self, fps):
                self.fingerprints = fps
                self.action_history = []  # no simple reversal can match
                self.metrics = []

        class R:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        cfg = {
            "oscillation_strategy": ["SIMPLE_REVERSAL", "STATE_FINGERPRINTING"],
            "fingerprint_exclusion_window": 0,
        }
        gh = GH(["z", "x"])  # contains 'x'
        ok, reason = detect_oscillation(gh, G("x"), R("A"), {}, cfg)
        assert ok and reason.startswith("[STATE_FINGERPRINTING]")

    def test_aggregator_all_no_detection(self):
        class G:
            def compute_fingerprint(self):
                return "nope"

        class GH:
            def __init__(self):
                self.fingerprints = ["a", "b"]
                self.action_history = [{"rule_info": ("X", None)}]
                self.metrics = [
                    {"num_nodes": 1, "num_edges": 1, "rule_applied": 0}
                    for _ in range(5)
                ]

        class R:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        cfg = {
            "oscillation_strategy": "ALL",
            "rule_reverse_pairs": {"A": "B"},
            "oscillation_history_window": 2,
            "oscillation_check_depth": 5,
            "fingerprint_exclusion_window": 1,
        }
        ok, reason = detect_oscillation(GH(), G(), R("C"), {}, cfg)
        assert ok is False and reason is None

    def test_simple_reversal_other_direction_mapping(self):
        class R:
            def __init__(self, rt):
                self.metadata = {"rule_type": rt}

        # Pair maps B->A; current A should reverse prior B
        history = [{"rule_info": ("B", None)}]
        cfg = {"rule_reverse_pairs": {"X": "Y", "B": "A"}, "oscillation_history_window": 3}
        ok, reason = detect_simple_reversal(R("A"), history, cfg)
        assert ok and "reverses previous" in reason
