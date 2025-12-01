"""Tests for tree deduction."""

import pytest
import math

from _hidden_tree import HiddenTree
from _deduction import (
    generate_problem,
    generate_observations,
    evaluate_answer,
    compute_ambiguity,
)
from _deduction_task import TreeDeductionTask


class TestObservations:
    """Test observation generation."""

    def test_reveal_fraction_1(self):
        """reveal_fraction=1 reveals all edges."""
        tree = HiddenTree(n=8, seed=42)
        obs = generate_observations(tree, reveal_fraction=1.0, seed=0)

        assert len(obs.revealed_edges) == 7  # n-1 edges
        assert obs.hidden_nodes == []

    def test_reveal_fraction_0(self):
        """reveal_fraction=0 reveals no edges."""
        tree = HiddenTree(n=8, seed=42)
        obs = generate_observations(tree, reveal_fraction=0.0, seed=0)

        assert len(obs.revealed_edges) == 0
        assert len(obs.hidden_nodes) == 7

    def test_reveal_fraction_half(self):
        """reveal_fraction=0.5 reveals about half."""
        tree = HiddenTree(n=10, seed=42)
        obs = generate_observations(tree, reveal_fraction=0.5, seed=0)

        # Should be approximately half (rounding)
        assert 3 <= len(obs.revealed_edges) <= 6

    def test_depths_always_revealed(self):
        """Depths are always revealed regardless of fraction."""
        tree = HiddenTree(n=8, seed=42)

        for frac in [0.0, 0.5, 1.0]:
            obs = generate_observations(tree, reveal_fraction=frac, seed=0)
            assert obs.depths == tree.depth

    def test_deterministic(self):
        """Same seeds produce same observations."""
        tree = HiddenTree(n=8, seed=42)

        obs1 = generate_observations(tree, 0.5, seed=123)
        obs2 = generate_observations(tree, 0.5, seed=123)

        assert [e.child for e in obs1.revealed_edges] == [e.child for e in obs2.revealed_edges]

    def test_candidate_parents(self):
        """Candidate parents are at depth-1."""
        tree = HiddenTree(n=8, seed=42)
        obs = generate_observations(tree, 0.0, seed=0)

        for node in obs.hidden_nodes:
            candidates = obs.candidate_parents(node)
            node_depth = obs.depths[node]

            for c in candidates:
                assert obs.depths[c] == node_depth - 1


class TestAmbiguity:
    """Test ambiguity computation."""

    def test_full_reveal_zero_ambiguity(self):
        """Full reveal has zero ambiguity."""
        tree = HiddenTree(n=8, seed=42)
        obs = generate_observations(tree, reveal_fraction=1.0, seed=0)

        assert compute_ambiguity(obs) == 0.0

    def test_no_reveal_positive_ambiguity(self):
        """No reveal has positive ambiguity."""
        tree = HiddenTree(n=8, seed=42)
        obs = generate_observations(tree, reveal_fraction=0.0, seed=0)

        assert compute_ambiguity(obs) > 0.0

    def test_ambiguity_decreases_with_reveal(self):
        """More reveal = less ambiguity."""
        tree = HiddenTree(n=10, seed=42)

        amb_0 = compute_ambiguity(generate_observations(tree, 0.0, seed=0))
        amb_5 = compute_ambiguity(generate_observations(tree, 0.5, seed=0))
        amb_1 = compute_ambiguity(generate_observations(tree, 1.0, seed=0))

        assert amb_0 >= amb_5 >= amb_1


class TestProblemGeneration:
    """Test problem generation."""

    def test_generate_problem(self):
        """Basic problem generation."""
        problem = generate_problem(n=8, reveal_fraction=0.5, seed=42)

        assert problem.n == 8
        assert problem.seed == 42
        assert problem.reveal_fraction == 0.5
        assert len(problem.ground_truth) == 8

    def test_deterministic_problems(self):
        """Same parameters produce same problem."""
        p1 = generate_problem(n=8, reveal_fraction=0.5, seed=42)
        p2 = generate_problem(n=8, reveal_fraction=0.5, seed=42)

        assert p1.ground_truth == p2.ground_truth
        assert p1.ambiguity_bits == p2.ambiguity_bits

    def test_prompt_generation(self):
        """Problem generates valid prompt."""
        problem = generate_problem(n=6, reveal_fraction=0.5, seed=42)
        prompt = problem.to_prompt()

        assert "Tree Deduction" in prompt
        assert "ANSWER" in prompt
        assert "depth" in prompt.lower()


class TestEvaluation:
    """Test answer evaluation."""

    def test_perfect_answer(self):
        """Correct answer gets score 1.0."""
        problem = generate_problem(n=8, reveal_fraction=0.5, seed=42)
        result = evaluate_answer(problem.ground_truth, problem)

        assert result["score"] == 1.0

    def test_wrong_answer(self):
        """Wrong answer gets lower score."""
        problem = generate_problem(n=8, reveal_fraction=0.0, seed=42)

        # All zeros (likely wrong)
        wrong = [-1] + [0] * 7
        result = evaluate_answer(wrong, problem)

        assert result["score"] < 1.0

    def test_only_hidden_scored(self):
        """Only hidden nodes count toward score."""
        problem = generate_problem(n=8, reveal_fraction=0.5, seed=42)

        # Copy ground truth
        answer = problem.ground_truth.copy()

        # Corrupt a revealed edge (shouldn't affect score)
        for obs in problem.observations.revealed_edges:
            answer[obs.child] = (obs.parent + 1) % problem.n
            break

        result = evaluate_answer(answer, problem)
        # Score based only on hidden nodes
        assert result["total"] == len(problem.observations.hidden_nodes)


class TestTask:
    """Test the task interface."""

    @pytest.fixture
    def task(self):
        return TreeDeductionTask(n=8, reveal_fraction=0.5)

    @pytest.mark.asyncio
    async def test_generate(self, task):
        """Task generates problem."""
        problem = await task.generate(seed=42)
        assert problem.n == 8

    @pytest.mark.asyncio
    async def test_evaluate_correct(self, task):
        """Correct answer scores 1.0."""
        problem = await task.generate(seed=42)
        gt = problem.ground_truth[1:]  # Skip root
        response = f"ANSWER: {' '.join(map(str, gt))}"

        score = await task.evaluate(response, problem)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_parse_formats(self, task):
        """Various answer formats are parsed."""
        problem = await task.generate(seed=42)
        gt = problem.ground_truth[1:]

        formats = [
            f"ANSWER: {' '.join(map(str, gt))}",
            f"SUBMIT: {','.join(map(str, gt))}",
            f"Solution: {' '.join(map(str, gt))}",
            f"[{', '.join(map(str, gt))}]",
        ]

        for response in formats:
            score = await task.evaluate(response, problem)
            assert score == 1.0, f"Failed: {response}"


class TestScaling:
    """Test scaling behavior."""

    def test_ambiguity_scales_with_n(self):
        """Larger trees have more ambiguity."""
        amb_small = compute_ambiguity(
            generate_observations(HiddenTree(5, 0), 0.0, 0)
        )
        amb_large = compute_ambiguity(
            generate_observations(HiddenTree(15, 0), 0.0, 0)
        )

        assert amb_large > amb_small

    def test_many_seeds(self):
        """Different seeds produce different problems."""
        problems = [generate_problem(8, 0.5, seed=i) for i in range(20)]
        trees = [tuple(p.ground_truth) for p in problems]

        # Should have variety
        assert len(set(trees)) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
