"""Tree Deduction Task - Clean single-turn evaluation.

Two parameters: n (size), reveal_fraction (difficulty).
That's it.
"""

import re
import random
from typing import Optional, List

from _deduction import generate_problem, evaluate_answer, DeductionProblem


class TreeDeductionTask:
    """Single-turn tree deduction from partial observations.

    Parameters:
        n: Number of nodes (default 8)
        reveal_fraction: Fraction of edges revealed (default 0.5)
            - 1.0 = trivial (all edges given)
            - 0.5 = moderate (half edges, deduce rest)
            - 0.0 = hard (only depths, deduce all edges)
    """

    # Pattern to extract answer
    ANSWER_PATTERN = re.compile(
        r'(?:ANSWER|SUBMIT|SOLUTION)[:\s]*([\d\s,\-]+)',
        re.IGNORECASE
    )

    def __init__(self, n: int = 8, reveal_fraction: float = 0.5):
        self.n = n
        self.reveal_fraction = reveal_fraction

    async def generate(
        self,
        n: Optional[int] = None,
        reveal_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        task_id: Optional[int] = None,
    ) -> DeductionProblem:
        """Generate a problem instance."""
        n = n if n is not None else self.n
        reveal_fraction = reveal_fraction if reveal_fraction is not None else self.reveal_fraction
        seed = seed if seed is not None else (task_id if task_id is not None else random.randint(0, 2**63 - 1))

        return generate_problem(n, reveal_fraction, seed)

    async def evaluate(self, response: str, problem: DeductionProblem) -> float:
        """Evaluate response, return score."""
        answer = self._parse(response, problem.n)
        if answer is None:
            return 0.0
        return evaluate_answer(answer, problem)["score"]

    def evaluate_detailed(self, response: str, problem: DeductionProblem) -> dict:
        """Evaluate with full details."""
        answer = self._parse(response, problem.n)
        if answer is None:
            return {"score": 0.0, "parse_error": True}
        result = evaluate_answer(answer, problem)
        result["predicted"] = answer
        return result

    def _parse(self, response: str, n: int) -> Optional[List[int]]:
        """Extract parent array from response."""
        # Try explicit pattern
        match = self.ANSWER_PATTERN.search(response)
        if match:
            values = [int(x) for x in re.findall(r'-?\d+', match.group(1))]
            return self._normalize(values, n)

        # Try bracketed array
        for m in re.finditer(r'\[([^\]]+)\]', response):
            values = [int(x) for x in re.findall(r'-?\d+', m.group(1))]
            if len(values) >= n - 1:
                return self._normalize(values, n)

        # Last line with enough numbers
        for line in reversed(response.strip().split('\n')):
            values = [int(x) for x in re.findall(r'-?\d+', line)]
            if len(values) >= n - 1:
                return self._normalize(values, n)

        return None

    def _normalize(self, values: List[int], n: int) -> List[int]:
        """Normalize to length-n array with root = -1."""
        if len(values) == n - 1:
            return [-1] + values
        if len(values) == n:
            return values
        if len(values) > n:
            return values[:n]
        return values + [-1] * (n - len(values))
