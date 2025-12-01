"""Tree Environment Actor.

This module provides the main Actor class for tree tasks,
following the affinetes pattern.

Supported tasks:
1. Tree Deduction (recommended): Single-turn reasoning from partial observations
2. Tree Reconstruction: Multi-turn interactive querying (legacy)
"""

import os
import time
import gc
import random
import re
import sys
import httpx
import openai

# Add /app to path for container imports
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from _task import TreeReconstructionTask
from _deduction_task import TreeDeductionTask
from _session import get_session_manager
from _models import Challenge


class Actor:
    """Tree Reconstruction evaluation actor.

    This actor supports evaluation in two modes:

    1. Single-turn mode (default):
       - Generate challenge prompt
       - Agent outputs queries and submission in one response
       - Parse and execute queries, evaluate submission

    2. Multi-turn mode:
       - Generate challenge prompt
       - Agent and environment exchange messages
       - Agent terminates by submitting reconstruction

    The multi-turn mode is more natural for this task but requires
    special handling in the evaluation loop.
    """

    # Maximum turns for multi-turn evaluation
    MAX_TURNS = 100

    # Patterns for parsing
    QUERY_PATTERN = re.compile(
        r'QUERY\s+(ANCESTOR|LCA|DEPTH|CHILDREN|PATH)\s+(\d+)(?:\s+(\d+))?',
        re.IGNORECASE
    )
    SUBMIT_PATTERN = re.compile(
        r'SUBMIT\s+([\d\s,\-]+)',
        re.IGNORECASE
    )

    def __init__(self, api_key: str = None):
        """Initialize Actor.

        Args:
            api_key: API key for LLM service. Uses CHUTES_API_KEY env var if not provided.
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.session_manager = get_session_manager()

    async def _llm_chat(
        self,
        messages,
        model,
        base_url,
        timeout,
        temperature,
        current_api_key,
        seed=None
    ):
        """Call LLM API with conversation history."""
        # Unset SSL_CERT_FILE to avoid certificate issues in container
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('REQUESTS_CA_BUNDLE', None)

        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip('/'),
            api_key=current_api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0
        )

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        if seed is not None:
            params["seed"] = seed

        response = await client.chat.completions.create(**params)

        if not response.choices:
            raise ValueError("LLM API returned empty choices list")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM API returned None content")

        return content.strip()

    async def evaluate(
        self,
        n: int = 20,
        method: str = "prufer",
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 600,
        temperature: float = 0.7,
        api_key: str = None,
        seed: int = None,
        task_id: int = None,
        max_queries: int = None,
        allowed_query_types: list = None,
        multi_turn: bool = True
    ):
        """Run tree reconstruction evaluation.

        Args:
            n: Number of nodes in the tree
            method: Generation method ("prufer" for uniform, "recursive")
            model: Model name for evaluation
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API calls
            temperature: Temperature for generation
            api_key: Override API key
            seed: Random seed for LLM generation
            task_id: Task ID for deterministic tree generation
            max_queries: Optional query limit
            allowed_query_types: Restrict query types (None = all)
            multi_turn: If True, run multi-turn loop; if False, single response

        Returns:
            Evaluation result dictionary
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        current_api_key = api_key or self.api_key

        # Create task with configuration
        task = TreeReconstructionTask(
            default_n=n,
            default_method=method,
            max_queries=max_queries,
            allowed_query_types=allowed_query_types
        )

        start = time.time()

        # Generate challenge
        challenge = await task.generate(n=n, seed=task_id, method=method, task_id=task_id)

        conversation = [{"role": "user", "content": challenge.prompt}]

        try:
            if multi_turn:
                score, final_result = await self._run_multi_turn(
                    task, challenge, conversation,
                    model, base_url, timeout, temperature, current_api_key, seed
                )
            else:
                score, final_result = await self._run_single_turn(
                    task, challenge, conversation,
                    model, base_url, timeout, temperature, current_api_key, seed
                )
            error = None
        except Exception as e:
            import traceback
            score = 0.0
            final_result = None
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        result = {
            "task_name": "tree:reconstruction",
            "score": score,
            "success": score > 0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "task_id": task_id,
                "n": n,
                "method": method,
                "evaluation": final_result
            }
        }

        if error:
            result["error"] = error
            result["error_type"] = "evaluation_failure"

        # Cleanup
        try:
            self.session_manager.close_session(challenge.session_id)
        except:
            pass

        gc.collect()
        return result

    async def _run_multi_turn(
        self,
        task,
        challenge,
        conversation,
        model,
        base_url,
        timeout,
        temperature,
        api_key,
        seed
    ):
        """Run multi-turn interactive evaluation."""
        for turn in range(self.MAX_TURNS):
            # Get LLM response
            response = await self._llm_chat(
                conversation, model, base_url, timeout, temperature, api_key, seed
            )
            conversation.append({"role": "assistant", "content": response})

            # Process response (may contain multiple queries + submission)
            result_messages = []
            is_complete = False
            final_result = None

            # Process all queries in the response
            for match in self.QUERY_PATTERN.finditer(response):
                query_type = match.group(1).upper()
                args = [int(match.group(2))]
                if match.group(3):
                    args.append(int(match.group(3)))

                try:
                    query_result = self.session_manager.query(
                        challenge.session_id, query_type, args
                    )
                    if query_type == "ANCESTOR":
                        answer = "YES" if query_result["result"] else "NO"
                    elif query_type in ("CHILDREN", "PATH"):
                        answer = "[" + ", ".join(map(str, query_result["result"])) + "]"
                    else:
                        answer = str(query_result["result"])
                    result_messages.append(f"{query_type} {' '.join(map(str, args))}: {answer}")
                except ValueError as e:
                    result_messages.append(f"Error: {e}")

            # Check for submission
            submit_match = self.SUBMIT_PATTERN.search(response)
            if submit_match:
                msg, is_complete, final_result = await task.process_response(
                    challenge.session_id, response
                )
                result_messages.append(msg)

            if result_messages:
                feedback = "\n".join(result_messages)
                conversation.append({"role": "user", "content": feedback})

            if is_complete:
                return final_result.get("score", 0.0) if final_result else 0.0, final_result

        # Timeout - max turns reached
        return 0.0, {"error": "max_turns_reached"}

    async def _run_single_turn(
        self,
        task,
        challenge,
        conversation,
        model,
        base_url,
        timeout,
        temperature,
        api_key,
        seed
    ):
        """Run single-turn evaluation (agent must solve in one response)."""
        # Get single LLM response
        response = await self._llm_chat(
            conversation, model, base_url, timeout, temperature, api_key, seed
        )
        conversation.append({"role": "assistant", "content": response})

        # Process all queries first
        for match in self.QUERY_PATTERN.finditer(response):
            query_type = match.group(1).upper()
            args = [int(match.group(2))]
            if match.group(3):
                args.append(int(match.group(3)))

            try:
                self.session_manager.query(challenge.session_id, query_type, args)
            except ValueError:
                pass  # Ignore errors in single-turn mode

        # Evaluate submission
        score = await task.evaluate(response, challenge)

        # Get session state for detailed result
        try:
            state = self.session_manager.get_session_state(challenge.session_id)
            final_result = {
                "score": score,
                "query_count": state.get("query_count", 0),
                "total_bits": state.get("total_bits", 0),
                "status": state.get("status", "unknown")
            }
        except:
            final_result = {"score": score}

        return score, final_result

    async def create_session(
        self,
        n: int = 20,
        seed: int = None,
        method: str = "prufer"
    ):
        """Create a new session for external interaction.

        This is useful for non-LLM agents or custom evaluation loops.

        Args:
            n: Number of nodes
            seed: Random seed (uses random if not provided)
            method: Generation method

        Returns:
            Dictionary with session_id and initial prompt
        """
        if seed is None:
            seed = random.randint(0, 2**63 - 1)

        task = TreeReconstructionTask(default_n=n, default_method=method)
        challenge = await task.generate(n=n, seed=seed, method=method)

        return {
            "session_id": challenge.session_id,
            "n": n,
            "seed": seed,
            "prompt": challenge.prompt,
            "info_lower_bound": challenge.extra["info_lower_bound"]
        }

    async def query(self, session_id: str, query_type: str, args: list):
        """Execute a query on an existing session.

        Args:
            session_id: Session identifier
            query_type: "ANCESTOR", "LCA", or "DEPTH"
            args: Query arguments

        Returns:
            Query result dictionary
        """
        return self.session_manager.query(session_id, query_type, args)

    async def submit(self, session_id: str, parent_array: list):
        """Submit a reconstruction for an existing session.

        Args:
            session_id: Session identifier
            parent_array: Parent array (length n, or n-1 without root)

        Returns:
            Evaluation result dictionary
        """
        # Handle both full array and array without root
        state = self.session_manager.get_session_state(session_id)
        n = state["n"]

        if len(parent_array) == n - 1:
            parent_array = [-1] + list(parent_array)

        return self.session_manager.submit(session_id, parent_array)

    async def get_ground_truth(self, session_id: str):
        """Get ground truth for a session (for testing/debugging).

        Args:
            session_id: Session identifier

        Returns:
            Ground truth parent array
        """
        if session_id not in self.session_manager.sessions:
            raise ValueError(f"Session not found: {session_id}")
        return self.session_manager.sessions[session_id].tree.get_ground_truth()

    # ==================== Deduction Task (Recommended) ====================

    async def evaluate_deduction(
        self,
        n: int = 8,
        reveal_fraction: float = 0.5,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 300,
        temperature: float = 0.7,
        api_key: str = None,
        seed: int = None,
        task_id: int = None,
    ):
        """Run tree deduction evaluation (single-turn).

        The agent receives partial observations about a hidden tree and must
        deduce the complete parent array. Tests constraint satisfaction and
        logical deduction.

        Args:
            n: Number of nodes (default 8)
            reveal_fraction: Fraction of edges revealed (default 0.5)
                - 1.0 = trivial (all edges given)
                - 0.5 = moderate (half edges, deduce rest)
                - 0.0 = hard (only depths given)
            model: Model name for evaluation
            base_url: Base URL for LLM API
            timeout: Timeout for LLM API call
            temperature: Temperature for generation
            api_key: Override API key
            seed: Random seed for LLM generation
            task_id: Task ID for deterministic generation

        Returns:
            Evaluation result dictionary
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        current_api_key = api_key or self.api_key
        actual_seed = task_id if task_id is not None else seed

        # Create task and problem
        task = TreeDeductionTask(n=n, reveal_fraction=reveal_fraction)
        problem = await task.generate(seed=actual_seed)

        start = time.time()

        try:
            messages = [{"role": "user", "content": problem.to_prompt()}]
            response = await self._llm_chat(
                messages, model, base_url, timeout, temperature, current_api_key, seed
            )

            detailed = task.evaluate_detailed(response, problem)
            score = detailed["score"]
            error = None

        except Exception as e:
            import traceback
            response = None
            score = 0.0
            detailed = {"error": str(e)}
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        result = {
            "task_name": "tree:deduction",
            "score": score,
            "success": score == 1.0,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": [
                    {"role": "user", "content": problem.to_prompt()},
                    {"role": "assistant", "content": response}
                ],
                "seed": actual_seed,
                "n": n,
                "reveal_fraction": reveal_fraction,
                "ambiguity_bits": problem.ambiguity_bits,
                "evaluation": detailed,
                "ground_truth": problem.ground_truth,
            }
        }

        if error:
            result["error"] = error
            result["error_type"] = "evaluation_failure"

        gc.collect()
        return result
