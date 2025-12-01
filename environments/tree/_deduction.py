"""Tree Deduction - Principled partial observation model.

The task: Given partial observations about a hidden tree, deduce the parent array.

Design Principle:
- Two parameters only: n (size) and reveal_fraction (difficulty)
- reveal_fraction ∈ [0, 1] controls what fraction of edges are directly revealed
- Depths are always given (makes the problem tractable)
- No arbitrary presets - difficulty emerges from the math

Information Theory:
- With depths known, each node at depth d has |{nodes at depth d-1}| parent choices
- reveal_fraction=1.0: all edges given, trivial (just read the answer)
- reveal_fraction=0.0: only depths given, must deduce from structure alone
- reveal_fraction=0.5: half edges given, deduce the rest

The elegance: difficulty scales continuously and meaningfully.
"""

import random
import math
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict

from _hidden_tree import HiddenTree


@dataclass
class Observation:
    """A single observation: (child, parent) edge."""
    child: int
    parent: int

    def __str__(self):
        return f"parent({self.child}) = {self.parent}"


@dataclass
class TreeObservations:
    """Complete observation set for a tree deduction problem."""
    n: int
    depths: List[int]  # depths[i] = depth of node i
    revealed_edges: List[Observation]  # Known (child, parent) pairs

    @property
    def revealed_fraction(self) -> float:
        """Fraction of edges revealed."""
        return len(self.revealed_edges) / (self.n - 1) if self.n > 1 else 1.0

    @property
    def hidden_nodes(self) -> List[int]:
        """Nodes whose parents are not revealed."""
        revealed = {obs.child for obs in self.revealed_edges}
        return [i for i in range(1, self.n) if i not in revealed]

    def nodes_at_depth(self, d: int) -> List[int]:
        """Get all nodes at a given depth."""
        return [i for i, depth in enumerate(self.depths) if depth == d]

    def candidate_parents(self, node: int) -> List[int]:
        """Get valid parent candidates for a node (nodes at depth-1)."""
        d = self.depths[node]
        if d == 0:
            return []  # Root has no parent
        return self.nodes_at_depth(d - 1)

    def to_prompt(self) -> str:
        """Convert to natural language prompt."""
        lines = [
            f"A tree has {self.n} nodes labeled 0 to {self.n-1}.",
            f"Node 0 is the root.",
            "",
            "Node depths:",
        ]

        # Group by depth for clarity
        by_depth = defaultdict(list)
        for i, d in enumerate(self.depths):
            by_depth[d].append(i)

        for d in sorted(by_depth.keys()):
            nodes = by_depth[d]
            lines.append(f"  Depth {d}: {', '.join(map(str, nodes))}")

        if self.revealed_edges:
            lines.append("")
            lines.append("Known edges:")
            for obs in self.revealed_edges:
                lines.append(f"  {obs}")

        lines.append("")
        lines.append(f"Hidden: parents of nodes {self.hidden_nodes}")

        return "\n".join(lines)


def generate_observations(
    tree: HiddenTree,
    reveal_fraction: float,
    seed: Optional[int] = None
) -> TreeObservations:
    """Generate partial observations for a tree.

    Args:
        tree: The hidden tree
        reveal_fraction: Fraction of edges to reveal (0 to 1)
        seed: Random seed for selecting which edges to reveal

    Returns:
        TreeObservations with depths and revealed edges
    """
    if not 0 <= reveal_fraction <= 1:
        raise ValueError(f"reveal_fraction must be in [0, 1], got {reveal_fraction}")

    rng = random.Random(seed)

    # Always reveal all depths
    depths = tree.depth.copy()

    # Select which edges to reveal
    all_edges = list(range(1, tree.n))  # Nodes 1 to n-1 (not root)
    num_to_reveal = int(round(reveal_fraction * len(all_edges)))

    revealed_nodes = set(rng.sample(all_edges, num_to_reveal)) if num_to_reveal > 0 else set()

    revealed_edges = [
        Observation(child=i, parent=tree.parent[i])
        for i in sorted(revealed_nodes)
    ]

    return TreeObservations(
        n=tree.n,
        depths=depths,
        revealed_edges=revealed_edges
    )


def count_consistent_trees(obs: TreeObservations) -> int:
    """Count trees consistent with observations.

    Uses dynamic programming over depth levels.
    Warning: Can be exponential, use only for small problems.
    """
    if obs.n > 12:
        raise ValueError("Counting only supported for n <= 12")

    # Build constraints from revealed edges
    known_parent: Dict[int, int] = {e.child: e.parent for e in obs.revealed_edges}

    # For each hidden node, get candidate parents
    hidden = obs.hidden_nodes
    candidates = {node: obs.candidate_parents(node) for node in hidden}

    # Count by enumeration
    def count_assignments(idx: int) -> int:
        if idx == len(hidden):
            return 1

        node = hidden[idx]
        total = 0
        for parent in candidates[node]:
            # This is a valid choice, recurse
            total += count_assignments(idx + 1)
        return total

    return count_assignments(0)


def compute_ambiguity(obs: TreeObservations) -> float:
    """Compute log2 of number of consistent trees (ambiguity in bits).

    This measures how much uncertainty remains after observations.
    - 0 bits = observations uniquely determine the tree
    - Higher = more ambiguous
    """
    # For each hidden node, count candidate parents
    total_bits = 0.0
    for node in obs.hidden_nodes:
        num_candidates = len(obs.candidate_parents(node))
        if num_candidates > 0:
            total_bits += math.log2(num_candidates)
    return total_bits


@dataclass
class DeductionProblem:
    """A complete tree deduction problem instance."""
    n: int
    seed: int
    reveal_fraction: float
    observations: TreeObservations
    ground_truth: List[int]  # The actual parent array
    ambiguity_bits: float  # Remaining uncertainty

    def to_prompt(self) -> str:
        """Generate the full problem prompt."""
        obs_text = self.observations.to_prompt()
        hidden = self.observations.hidden_nodes

        return f"""# Tree Deduction

{obs_text}

## Task

Determine the parent of each hidden node: {hidden}

Constraints:
- A node at depth d must have a parent at depth d-1
- Use the known edges and depth structure to deduce the hidden edges

## Answer Format

ANSWER: p1 p2 p3 ... p{self.n - 1}

Where pi is the parent of node i (for i = 1 to {self.n - 1}).
"""


def generate_problem(
    n: int,
    reveal_fraction: float,
    seed: int
) -> DeductionProblem:
    """Generate a complete deduction problem.

    Args:
        n: Number of nodes
        reveal_fraction: Fraction of edges to reveal (0=hard, 1=trivial)
        seed: Random seed for reproducibility

    Returns:
        DeductionProblem instance
    """
    # Generate tree
    tree = HiddenTree(n, seed, method="prufer")

    # Generate observations with derived seed
    obs = generate_observations(tree, reveal_fraction, seed=seed + 1_000_000)

    # Compute ambiguity
    ambiguity = compute_ambiguity(obs)

    return DeductionProblem(
        n=n,
        seed=seed,
        reveal_fraction=reveal_fraction,
        observations=obs,
        ground_truth=tree.parent,
        ambiguity_bits=ambiguity
    )


def evaluate_answer(answer: List[int], problem: DeductionProblem) -> dict:
    """Evaluate a proposed parent array.

    Args:
        answer: Proposed parent array (length n)
        problem: The deduction problem

    Returns:
        Dict with score and details
    """
    if len(answer) != problem.n:
        return {
            "score": 0.0,
            "error": f"Expected {problem.n} values, got {len(answer)}"
        }

    # Check only the hidden nodes (revealed ones are given)
    hidden = problem.observations.hidden_nodes
    correct = sum(
        1 for node in hidden
        if answer[node] == problem.ground_truth[node]
    )

    total = len(hidden)
    score = correct / total if total > 0 else 1.0

    errors = [
        {"node": node, "predicted": answer[node], "actual": problem.ground_truth[node]}
        for node in hidden
        if answer[node] != problem.ground_truth[node]
    ]

    return {
        "score": score,
        "correct": correct,
        "total": total,
        "errors": errors,
        "hidden_nodes": hidden
    }
