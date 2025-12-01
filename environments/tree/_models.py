"""Data models for tree reconstruction challenges and evaluations."""

import time
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class Challenge(BaseModel):
    """Challenge specification for evaluation.

    This follows the affinetes Challenge pattern for compatibility.
    """

    env: str
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[float] = Field(default_factory=lambda: time.time())


class TreeChallenge(Challenge):
    """Extended challenge for tree reconstruction with session tracking."""

    session_id: str
    n: int
    seed: int


class QueryResult(BaseModel):
    """Result of a tree query."""

    query_type: str
    args: List[int]
    result: Any
    bits: float
    total_queries: int
    total_bits: float
    info_lower_bound: float


class EvaluationResult(BaseModel):
    """Result of evaluating a tree reconstruction."""

    score: float
    correct: int
    total: int
    query_count: int
    total_bits: float
    info_lower_bound: float
    efficiency: Optional[float]
    status: str


class SessionSummary(BaseModel):
    """Summary of a completed session."""

    session_id: str
    n: int
    seed: int
    method: str
    score: float
    query_count: int
    total_bits: float
    info_lower_bound: float
    efficiency: Optional[float]
    conversation: List[Dict[str, str]]
