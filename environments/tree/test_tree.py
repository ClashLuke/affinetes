"""Unit tests for tree reconstruction environment.

Tests cover:
1. Tree generation (Prüfer and recursive)
2. Query oracles (ANCESTOR, LCA, DEPTH)
3. Session management
4. Evaluation scoring
5. Information accounting
"""

import pytest
import asyncio
import random
import math
from collections import deque

from _hidden_tree import HiddenTree, QueryResult, TreeStats
from _session import SessionManager, SessionStatus
from _task import TreeReconstructionTask


class TestHiddenTree:
    """Tests for HiddenTree class."""

    def test_basic_creation(self):
        """Test basic tree creation."""
        tree = HiddenTree(n=10, seed=42)
        assert tree.n == 10
        assert tree.parent[0] == -1  # Root has no parent
        assert len(tree.parent) == 10

    def test_deterministic_generation(self):
        """Test that same seed produces same tree."""
        tree1 = HiddenTree(n=20, seed=12345)
        tree2 = HiddenTree(n=20, seed=12345)
        assert tree1.parent == tree2.parent

    def test_different_seeds_different_trees(self):
        """Test that different seeds produce different trees (with high probability)."""
        trees = [HiddenTree(n=20, seed=i) for i in range(10)]
        parent_arrays = [tuple(t.parent) for t in trees]
        # All should be unique
        assert len(set(parent_arrays)) == 10

    def test_prufer_vs_recursive(self):
        """Test both generation methods work."""
        tree_prufer = HiddenTree(n=15, seed=42, method="prufer")
        tree_recursive = HiddenTree(n=15, seed=42, method="recursive")

        # Both should be valid trees
        assert tree_prufer.parent[0] == -1
        assert tree_recursive.parent[0] == -1

        # They will likely be different (different distributions)
        # Just verify they're both valid
        for tree in [tree_prufer, tree_recursive]:
            for i in range(1, tree.n):
                assert 0 <= tree.parent[i] < tree.n

    def test_tree_connectivity(self):
        """Test that generated tree is connected."""
        for seed in range(10):
            tree = HiddenTree(n=20, seed=seed)

            # BFS from root should reach all nodes
            visited = {0}
            queue = deque([0])
            while queue:
                u = queue.popleft()
                for v in tree.children[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)

            assert len(visited) == tree.n, f"Tree disconnected with seed {seed}"

    def test_small_trees(self):
        """Test edge cases with small trees."""
        tree2 = HiddenTree(n=2, seed=0)
        assert tree2.parent == [-1, 0]

        tree3 = HiddenTree(n=3, seed=42)
        assert tree3.parent[0] == -1
        assert 0 <= tree3.parent[1] < 3
        assert 0 <= tree3.parent[2] < 3


class TestQueryOracles:
    """Tests for query oracles."""

    @pytest.fixture
    def sample_tree(self):
        """Create a sample tree for testing."""
        # Use a fixed seed for deterministic tests
        return HiddenTree(n=10, seed=42)

    def test_ancestor_self(self, sample_tree):
        """Test that a node is its own ancestor."""
        for i in range(sample_tree.n):
            result = sample_tree.query_ancestor(i, i)
            assert result.result is True

    def test_ancestor_root(self, sample_tree):
        """Test that root is ancestor of all nodes."""
        for i in range(sample_tree.n):
            result = sample_tree.query_ancestor(0, i)
            assert result.result is True

    def test_ancestor_parent(self, sample_tree):
        """Test that parent is ancestor of child."""
        for i in range(1, sample_tree.n):
            parent = sample_tree.parent[i]
            result = sample_tree.query_ancestor(parent, i)
            assert result.result is True

    def test_ancestor_non_ancestor(self, sample_tree):
        """Test that non-ancestors return False."""
        # Find two nodes at same depth (siblings or cousins)
        # They should not be ancestors of each other
        depths = {}
        for i in range(sample_tree.n):
            d = sample_tree.depth[i]
            if d not in depths:
                depths[d] = []
            depths[d].append(i)

        # Find depth with multiple nodes
        for d, nodes in depths.items():
            if len(nodes) >= 2:
                u, v = nodes[0], nodes[1]
                if u != v:
                    result_uv = sample_tree.query_ancestor(u, v)
                    result_vu = sample_tree.query_ancestor(v, u)
                    # At same depth, neither is ancestor of other (unless same node)
                    assert not (result_uv.result and result_vu.result)
                    break

    def test_lca_same_node(self, sample_tree):
        """Test LCA of a node with itself."""
        for i in range(sample_tree.n):
            result = sample_tree.query_lca(i, i)
            assert result.result == i

    def test_lca_with_root(self, sample_tree):
        """Test LCA with root is always root."""
        for i in range(sample_tree.n):
            result = sample_tree.query_lca(0, i)
            assert result.result == 0

    def test_lca_symmetry(self, sample_tree):
        """Test LCA is symmetric."""
        for _ in range(20):
            u = random.randint(0, sample_tree.n - 1)
            v = random.randint(0, sample_tree.n - 1)
            lca_uv = sample_tree.query_lca(u, v).result
            lca_vu = sample_tree.query_lca(v, u).result
            assert lca_uv == lca_vu

    def test_lca_is_ancestor_of_both(self, sample_tree):
        """Test that LCA is ancestor of both inputs."""
        for _ in range(20):
            u = random.randint(0, sample_tree.n - 1)
            v = random.randint(0, sample_tree.n - 1)
            lca = sample_tree.query_lca(u, v).result
            assert sample_tree.query_ancestor(lca, u).result
            assert sample_tree.query_ancestor(lca, v).result

    def test_depth_root(self, sample_tree):
        """Test depth of root is 0."""
        result = sample_tree.query_depth(0)
        assert result.result == 0

    def test_depth_consistency(self, sample_tree):
        """Test depth is parent's depth + 1."""
        for i in range(1, sample_tree.n):
            depth_i = sample_tree.query_depth(i).result
            parent = sample_tree.parent[i]
            depth_parent = sample_tree.query_depth(parent).result
            assert depth_i == depth_parent + 1

    def test_query_bits(self, sample_tree):
        """Test information bits are correctly reported."""
        ancestor_result = sample_tree.query_ancestor(0, 1)
        assert ancestor_result.bits == 1.0

        lca_result = sample_tree.query_lca(0, 1)
        assert lca_result.bits == math.ceil(math.log2(sample_tree.n))

    def test_invalid_nodes(self, sample_tree):
        """Test that invalid nodes raise ValueError."""
        with pytest.raises(ValueError):
            sample_tree.query_ancestor(-1, 0)
        with pytest.raises(ValueError):
            sample_tree.query_ancestor(0, sample_tree.n)
        with pytest.raises(ValueError):
            sample_tree.query_lca(100, 0)


class TestEvaluation:
    """Tests for reconstruction evaluation."""

    def test_perfect_reconstruction(self):
        """Test perfect reconstruction gets score 1.0."""
        tree = HiddenTree(n=10, seed=42)
        result = tree.evaluate_reconstruction(tree.parent)
        assert result["score"] == 1.0
        assert result["correct"] == 9  # n-1 edges
        assert len(result["errors"]) == 0

    def test_completely_wrong(self):
        """Test completely wrong reconstruction."""
        tree = HiddenTree(n=10, seed=42)
        # All nodes point to root (likely wrong for most)
        wrong_parent = [-1] + [0] * 9
        result = tree.evaluate_reconstruction(wrong_parent)
        assert result["score"] < 1.0

    def test_partial_correctness(self):
        """Test partial reconstruction."""
        tree = HiddenTree(n=10, seed=42)
        # Copy and corrupt some entries
        partial = tree.parent.copy()
        partial[1] = (partial[1] + 1) % tree.n
        partial[2] = (partial[2] + 1) % tree.n

        result = tree.evaluate_reconstruction(partial)
        assert 0 < result["score"] < 1.0
        assert result["correct"] == 7  # 9 - 2 errors

    def test_wrong_length_raises(self):
        """Test wrong length parent array raises ValueError."""
        tree = HiddenTree(n=10, seed=42)
        with pytest.raises(ValueError):
            tree.evaluate_reconstruction([0] * 5)


class TestSessionManager:
    """Tests for session management."""

    @pytest.fixture
    def manager(self):
        return SessionManager(max_sessions=10)

    def test_create_session(self, manager):
        """Test session creation."""
        info = manager.create_session(n=15, seed=42)
        assert info.n == 15
        assert info.root == 0
        assert "ANCESTOR" in info.available_queries
        assert info.session_id in manager.sessions

    def test_session_determinism(self, manager):
        """Test that same seed creates equivalent sessions."""
        info1 = manager.create_session(n=10, seed=123)
        info2 = manager.create_session(n=10, seed=123)

        tree1 = manager.sessions[info1.session_id].tree
        tree2 = manager.sessions[info2.session_id].tree

        assert tree1.parent == tree2.parent

    def test_query_execution(self, manager):
        """Test query execution through manager."""
        info = manager.create_session(n=10, seed=42)

        result = manager.query(info.session_id, "ANCESTOR", [0, 5])
        assert "result" in result
        assert result["total_queries"] == 1

        result = manager.query(info.session_id, "LCA", [3, 7])
        assert result["total_queries"] == 2

    def test_bits_tracking(self, manager):
        """Test information bits are tracked."""
        info = manager.create_session(n=10, seed=42)

        manager.query(info.session_id, "ANCESTOR", [0, 1])
        state = manager.get_session_state(info.session_id)
        assert state["total_bits"] == 1.0

        manager.query(info.session_id, "LCA", [0, 1])
        state = manager.get_session_state(info.session_id)
        assert state["total_bits"] > 1.0

    def test_submit_reconstruction(self, manager):
        """Test submission and evaluation."""
        info = manager.create_session(n=10, seed=42)
        tree = manager.sessions[info.session_id].tree

        # Submit correct answer
        result = manager.submit(info.session_id, tree.parent)
        assert result["score"] == 1.0
        assert result["status"] == "submitted"

    def test_cannot_query_after_submit(self, manager):
        """Test that queries fail after submission."""
        info = manager.create_session(n=10, seed=42)
        tree = manager.sessions[info.session_id].tree

        manager.submit(info.session_id, tree.parent)

        with pytest.raises(ValueError):
            manager.query(info.session_id, "ANCESTOR", [0, 1])

    def test_session_not_found(self, manager):
        """Test error on invalid session ID."""
        with pytest.raises(ValueError):
            manager.query("nonexistent", "ANCESTOR", [0, 1])


class TestTreeReconstructionTask:
    """Tests for the task interface."""

    @pytest.fixture
    def task(self):
        return TreeReconstructionTask(default_n=10)

    @pytest.mark.asyncio
    async def test_generate_challenge(self, task):
        """Test challenge generation."""
        challenge = await task.generate(n=15, seed=42)
        assert challenge.env == "tree"
        assert challenge.n == 15
        assert challenge.seed == 42
        assert "QUERY" in challenge.prompt
        assert "SUBMIT" in challenge.prompt

    @pytest.mark.asyncio
    async def test_process_query(self, task):
        """Test processing query responses."""
        challenge = await task.generate(n=10, seed=42)

        msg, complete, result = await task.process_response(
            challenge.session_id,
            "Let me query: QUERY ANCESTOR 0 5"
        )
        assert not complete
        assert "YES" in msg or "NO" in msg

    @pytest.mark.asyncio
    async def test_process_submission(self, task):
        """Test processing submission."""
        challenge = await task.generate(n=5, seed=42)
        ground_truth = task.get_ground_truth(challenge.session_id)

        # Submit correct answer (skip root)
        submission = "SUBMIT " + " ".join(map(str, ground_truth[1:]))
        msg, complete, result = await task.process_response(
            challenge.session_id,
            submission
        )
        assert complete
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_single_turn(self, task):
        """Test single-turn evaluation."""
        challenge = await task.generate(n=5, seed=42)
        ground_truth = task.get_ground_truth(challenge.session_id)

        response = f"After analysis, I submit: SUBMIT {' '.join(map(str, ground_truth[1:]))}"
        score = await task.evaluate(response, challenge)
        assert score == 1.0


class TestInformationTheory:
    """Tests for information-theoretic properties."""

    def test_lower_bound_formula(self):
        """Test lower bound calculation."""
        for n in [10, 50, 100]:
            tree = HiddenTree(n=n, seed=42)
            lb = tree.get_theoretical_lower_bound()
            expected = (n - 1) * math.log2(n)
            assert abs(lb - expected) < 0.001

    def test_prufer_sequence_coverage(self):
        """Test that Prüfer generates diverse trees."""
        n = 8
        seen_trees = set()

        for seed in range(100):
            tree = HiddenTree(n=n, seed=seed, method="prufer")
            seen_trees.add(tuple(tree.parent))

        # Should see many different trees
        assert len(seen_trees) > 50

    def test_efficiency_calculation(self):
        """Test efficiency is calculated correctly."""
        manager = SessionManager()
        info = manager.create_session(n=10, seed=42)

        # Make some queries
        for i in range(5):
            manager.query(info.session_id, "ANCESTOR", [0, i + 1])

        state = manager.sessions[info.session_id]
        assert state.total_bits == 5.0
        expected_efficiency = state.info_lower_bound / 5.0
        assert abs(state.efficiency - expected_efficiency) < 0.001


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_min_tree_size(self):
        """Test minimum tree size."""
        tree = HiddenTree(n=2, seed=0)
        assert tree.parent == [-1, 0]

    def test_invalid_tree_size(self):
        """Test invalid tree size raises error."""
        with pytest.raises(ValueError):
            HiddenTree(n=1, seed=0)

    def test_invalid_method(self):
        """Test invalid generation method raises error."""
        with pytest.raises(ValueError):
            HiddenTree(n=10, seed=0, method="invalid")

    @pytest.mark.asyncio
    async def test_invalid_query_type(self):
        """Test invalid query type in task."""
        task = TreeReconstructionTask()
        challenge = await task.generate(n=10, seed=42)

        msg, complete, result = await task.process_response(
            challenge.session_id,
            "QUERY INVALID 0 1"
        )
        assert "Error" in msg or "Could not parse" in msg

    @pytest.mark.asyncio
    async def test_malformed_submission(self):
        """Test malformed submission handling."""
        task = TreeReconstructionTask()
        challenge = await task.generate(n=10, seed=42)

        msg, complete, result = await task.process_response(
            challenge.session_id,
            "SUBMIT abc def"
        )
        assert not complete or "Error" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
