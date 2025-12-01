"""Hidden tree with query oracles for tree reconstruction.

This module implements a principled tree reconstruction environment where:
- Trees are generated uniformly at random using Prüfer sequences
- Query oracles provide information about the hidden tree structure
- Information complexity is tracked for theoretical analysis

Information-Theoretic Foundation:
- There are n^(n-1) rooted labeled trees on n nodes (Cayley's formula)
- Lower bound: Ω(n log n) bits of information needed
- ANCESTOR queries provide 1 bit each
- LCA queries provide O(log n) bits each
"""

import random
import math
from typing import List, Tuple, Optional, Set
from collections import deque
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """Result of a query with information accounting."""
    result: any  # The actual query result
    bits: float  # Information bits revealed by this query
    query_type: str
    args: Tuple


@dataclass
class TreeStats:
    """Statistics about tree structure."""
    n: int
    height: int
    avg_depth: float
    max_branching: int
    leaf_count: int


class HiddenTree:
    """A hidden rooted tree with query oracles.

    The tree is generated deterministically from a seed using one of:
    - Prüfer sequence (uniform distribution over labeled trees)
    - Random recursive tree (simpler, non-uniform)

    Query Types:
    - ANCESTOR(u, v): Is u an ancestor of v? (1 bit)
    - LCA(u, v): Lowest common ancestor (ceil(log2(n)) bits)
    - DEPTH(v): Depth of node v (ceil(log2(height)) bits)
    """

    def __init__(self, n: int, seed: int, method: str = "prufer"):
        """Initialize a hidden tree.

        Args:
            n: Number of nodes (labeled 0 to n-1, root is always 0)
            seed: Random seed for deterministic generation
            method: "prufer" for uniform distribution, "recursive" for random recursive tree
        """
        if n < 2:
            raise ValueError(f"Tree must have at least 2 nodes, got {n}")

        self.n = n
        self.seed = seed
        self.method = method

        # Generate the tree
        rng = random.Random(seed)
        if method == "prufer":
            self.parent = self._generate_prufer(n, rng)
        elif method == "recursive":
            self.parent = self._generate_recursive(n, rng)
        else:
            raise ValueError(f"Unknown generation method: {method}")

        # Precompute depth and children for efficient queries
        self._precompute()

    def _generate_prufer(self, n: int, rng: random.Random) -> List[int]:
        """Generate a tree uniformly at random using Prüfer sequence.

        The Prüfer sequence establishes a bijection between labeled trees
        on n vertices and sequences of length n-2 with elements in {0, ..., n-1}.
        This gives uniform distribution over all labeled trees.

        Args:
            n: Number of nodes
            rng: Random number generator

        Returns:
            Parent array where parent[i] is the parent of node i, parent[0] = -1
        """
        if n == 2:
            # Special case: only one possible tree (0-1 edge)
            return [-1, 0]

        # Generate random Prüfer sequence of length n-2
        sequence = [rng.randrange(n) for _ in range(n - 2)]

        # Convert Prüfer sequence to edge list
        edges = self._prufer_to_edges(sequence, n)

        # Root the tree at node 0
        return self._root_tree(edges, root=0)

    def _prufer_to_edges(self, sequence: List[int], n: int) -> List[Tuple[int, int]]:
        """Convert Prüfer sequence to undirected edge list.

        Algorithm: Track degree of each node (1 + occurrences in sequence).
        Repeatedly connect smallest leaf to next sequence element.
        """
        # Count degree: each node starts with degree 1, sequence members get +1
        degree = [1] * n
        for node in sequence:
            degree[node] += 1

        edges = []
        ptr = 0

        # Find smallest leaf (smallest node with degree 1)
        while ptr < n and degree[ptr] != 1:
            ptr += 1
        leaf = ptr

        for node in sequence:
            edges.append((leaf, node))
            degree[leaf] -= 1
            degree[node] -= 1

            # If node became a leaf and is smaller than current pointer
            if degree[node] == 1 and node < ptr:
                leaf = node
            else:
                # Find next smallest leaf
                ptr += 1
                while ptr < n and degree[ptr] != 1:
                    ptr += 1
                leaf = ptr

        # Last edge connects the two remaining degree-1 nodes
        # One is leaf (ptr), find the other
        for i in range(n):
            if degree[i] == 1 and i != leaf:
                edges.append((leaf, i))
                break

        return edges

    def _root_tree(self, edges: List[Tuple[int, int]], root: int = 0) -> List[int]:
        """Convert undirected edges to parent array rooted at given node.

        Args:
            edges: List of undirected edges
            root: Root node (default 0)

        Returns:
            Parent array where parent[root] = -1
        """
        n = len(edges) + 1

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS from root to assign parents
        parent = [-1] * n
        visited = [False] * n
        queue = deque([root])
        visited[root] = True

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)

        return parent

    def _generate_recursive(self, n: int, rng: random.Random) -> List[int]:
        """Generate random recursive tree.

        For each node i (1 to n-1), attach it to a uniformly random
        node in {0, ..., i-1}. This produces a non-uniform distribution
        over labeled trees (favors more balanced structures).

        Args:
            n: Number of nodes
            rng: Random number generator

        Returns:
            Parent array
        """
        parent = [-1] * n  # parent[0] = -1 (root)
        for i in range(1, n):
            parent[i] = rng.randrange(i)  # Random parent from existing nodes
        return parent

    def _precompute(self):
        """Precompute auxiliary data structures for efficient queries."""
        n = self.n

        # Compute depth for each node
        self.depth = [0] * n
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in range(n):
                if self.parent[v] == u:
                    self.depth[v] = self.depth[u] + 1
                    queue.append(v)

        # Compute children lists
        self.children = [[] for _ in range(n)]
        for v in range(1, n):
            self.children[self.parent[v]].append(v)

        # Compute height (max depth)
        self.height = max(self.depth)

        # Information bits per query type
        self._bits_per_ancestor = 1.0
        self._bits_per_lca = math.ceil(math.log2(n)) if n > 1 else 1
        self._bits_per_depth = math.ceil(math.log2(self.height + 1)) if self.height > 0 else 1

    # ==================== Query Oracles ====================

    def query_ancestor(self, u: int, v: int) -> QueryResult:
        """Query: Is u an ancestor of v?

        Args:
            u, v: Node indices

        Returns:
            QueryResult with boolean result (True if u is ancestor of v)

        Note: A node is considered an ancestor of itself.
        """
        self._validate_nodes(u, v)

        # Walk up from v to root, checking if we encounter u
        result = self._is_ancestor(u, v)

        return QueryResult(
            result=result,
            bits=self._bits_per_ancestor,
            query_type="ANCESTOR",
            args=(u, v)
        )

    def query_lca(self, u: int, v: int) -> QueryResult:
        """Query: What is the lowest common ancestor of u and v?

        Args:
            u, v: Node indices

        Returns:
            QueryResult with LCA node index
        """
        self._validate_nodes(u, v)

        # Collect ancestors of u (including u)
        ancestors_u = set()
        current = u
        while current != -1:
            ancestors_u.add(current)
            current = self.parent[current]

        # Walk up from v until we find common ancestor
        current = v
        while current not in ancestors_u:
            current = self.parent[current]

        return QueryResult(
            result=current,
            bits=self._bits_per_lca,
            query_type="LCA",
            args=(u, v)
        )

    def query_depth(self, v: int) -> QueryResult:
        """Query: What is the depth of node v?

        Args:
            v: Node index

        Returns:
            QueryResult with depth (root has depth 0)
        """
        self._validate_nodes(v)

        return QueryResult(
            result=self.depth[v],
            bits=self._bits_per_depth,
            query_type="DEPTH",
            args=(v,)
        )

    def query_children(self, v: int) -> QueryResult:
        """Query: What are the children of node v?

        This is a more powerful query that reveals the entire child list.
        Useful for reducing the number of interaction turns.

        Args:
            v: Node index

        Returns:
            QueryResult with list of child node indices (may be empty for leaves)
        """
        self._validate_nodes(v)

        children = self.children[v].copy()
        # Information: we learn the exact set of children
        # Bits: log2(n choose k) where k = number of children, approximated as k * log2(n)
        num_children = len(children)
        bits = num_children * math.ceil(math.log2(self.n)) if num_children > 0 else 1.0

        return QueryResult(
            result=children,
            bits=bits,
            query_type="CHILDREN",
            args=(v,)
        )

    def query_path(self, u: int, v: int) -> QueryResult:
        """Query: What is the path from u to v?

        Returns the sequence of nodes on the path from u to v through their LCA.
        This is a very powerful query that can reveal multiple edges at once.

        Args:
            u, v: Node indices

        Returns:
            QueryResult with list of nodes on path from u to v (inclusive)
        """
        self._validate_nodes(u, v)

        # Find path u -> LCA
        path_u = []
        current = u
        ancestors_u = set()
        while current != -1:
            path_u.append(current)
            ancestors_u.add(current)
            current = self.parent[current]

        # Find path v -> LCA
        path_v = []
        current = v
        while current not in ancestors_u:
            path_v.append(current)
            current = self.parent[current]
        lca = current

        # Construct full path: u -> LCA -> v
        # path_u is [u, ..., LCA, ..., root], we need [u, ..., LCA]
        lca_idx = path_u.index(lca)
        path = path_u[:lca_idx + 1] + path_v[::-1]

        # Information: path length * log2(n) bits
        bits = len(path) * math.ceil(math.log2(self.n))

        return QueryResult(
            result=path,
            bits=bits,
            query_type="PATH",
            args=(u, v)
        )

    def _is_ancestor(self, u: int, v: int) -> bool:
        """Check if u is an ancestor of v (internal, no validation)."""
        current = v
        while current != -1:
            if current == u:
                return True
            current = self.parent[current]
        return False

    def _validate_nodes(self, *nodes):
        """Validate that all nodes are in valid range."""
        for node in nodes:
            if not (0 <= node < self.n):
                raise ValueError(f"Node {node} out of range [0, {self.n})")

    # ==================== Evaluation ====================

    def evaluate_reconstruction(self, predicted_parent: List[int]) -> dict:
        """Evaluate a predicted parent array against ground truth.

        Args:
            predicted_parent: Predicted parent array (length n, index 0 ignored)

        Returns:
            Dictionary with:
            - score: Accuracy (fraction of correct parent assignments)
            - correct: Number of correct assignments
            - total: Total assignments to check (n-1)
            - errors: List of incorrect (node, predicted, actual) tuples
        """
        if len(predicted_parent) != self.n:
            raise ValueError(f"Expected parent array of length {self.n}, got {len(predicted_parent)}")

        correct = 0
        errors = []

        for i in range(1, self.n):  # Skip root (index 0)
            if predicted_parent[i] == self.parent[i]:
                correct += 1
            else:
                errors.append((i, predicted_parent[i], self.parent[i]))

        total = self.n - 1
        score = correct / total if total > 0 else 1.0

        return {
            "score": score,
            "correct": correct,
            "total": total,
            "errors": errors
        }

    def get_stats(self) -> TreeStats:
        """Get statistics about the tree structure."""
        return TreeStats(
            n=self.n,
            height=self.height,
            avg_depth=sum(self.depth) / self.n,
            max_branching=max(len(c) for c in self.children),
            leaf_count=sum(1 for c in self.children if len(c) == 0)
        )

    def get_theoretical_lower_bound(self) -> float:
        """Get information-theoretic lower bound in bits.

        For n nodes, there are (n-1) parent choices, each from up to n-1 options,
        giving roughly n^(n-1) possible trees. The lower bound is:
        log2(n^(n-1)) = (n-1) * log2(n) bits
        """
        return (self.n - 1) * math.log2(self.n)

    def get_ground_truth(self) -> List[int]:
        """Return the ground truth parent array (for debugging/testing)."""
        return self.parent.copy()
