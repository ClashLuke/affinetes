"""Python interpreter simulation environment.

This environment generates short Python programs and asks the model to
predict the exact stdout of running them. The prompt follows the
same structure as `python_env.py` in the repo, but the implementation
is containerized so it can be built/run via `afs`.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import httpx
import openai

# Operations that the generator may use when building a program
DEFAULT_ALLOWED_OPS: tuple[str, ...] = (
    "PRINT",
    "SORT",
    "APPEND",
    "ADD",
    "ASSIGN",
    "NOOP",
    "REVERSE",
    "POP",
    "ZIP",
    "EXTEND",
    "INSERT",
    "MIN",
    "MAX",
)

PROMPT_PREAMBLE = (
    "In the following, we will test your ability to understand and execute python code.\n\n"
    "RULES:\n"
    "* You will not be able to use any external tools, such as a calculator or python during this test\n"
    "* You will see the python code precisely once\n"
    "* You may think and decypher the test for as long as you want\n"
    "* Return the *exact* printout the python interpreter would otherwise return, including formatting and potential typos in the challenge\n"
    "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n\n"
    "Example:\n"
    "CHALLENGE:\n"
    ">>> var1 = [62139]\n"
    ">>> var2 = [62598]\n"
    ">>> print(var1)\n"
    ">>> var4 = [40899]\n"
    ">>> var5 = sorted(var2)\n"
    ">>> var7 = 86093\n"
    ">>> var7 += 35501\n"
    ">>> print(var7)\n"
    "RESPONSE:\n"
    "Okay, I've processed the code you provided.\n"
    "<ANSWER>[62139]\n"
    "121594\n"
    "</ANSWER>\n\n"
    "Below, you will see the real task. Remember and follow the rules.\n\n"
    "CHALLENGE:\n"
)


def _flatten_keys(raw: Dict[str, Any], lists: Dict[str, list]) -> List[str]:
    """Return a flat list of variable names."""
    return list(raw.keys()) + list(lists.keys())


def build_program(
    seed: int,
    op_count: int = 10,
    allowed_ops: Sequence[str] = DEFAULT_ALLOWED_OPS,
    max_digits: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Generate a small Python program.

    Returns
    -------
    code_lines: list[str]
        Executable Python statements (without >>>).
    ops_used: list[str]
        Operations that actually emitted code.
    """
    rng = random.Random(seed)
    code_lines: list[str] = []
    ops_used: list[str] = []
    vars_state: dict[str, dict[str, Any]] = defaultdict(dict)
    counter = 0

    def new_name() -> str:
        nonlocal counter
        counter += 1
        return f"var{counter}"

    def rand_int() -> int:
        return rng.randint(0, 10**max_digits - 1)

    def pick(kind: str) -> str | None:
        keys = list(vars_state[kind].keys())
        return rng.choice(keys) if keys else None

    def pick_raw_value() -> tuple[str, int]:
        name = pick("raw")
        if name:
            return name, vars_state["raw"][name]
        value = rand_int()
        return str(value), value

    def op_print():
        target = rng.choice(_flatten_keys(vars_state["raw"], vars_state["list"])) if _flatten_keys(vars_state["raw"], vars_state["list"]) else None
        code_lines.append(f"print({target})" if target else "print()")
        ops_used.append("PRINT")

    def op_sort():
        source = pick("list")
        if not source:
            return
        target = new_name()
        vars_state["list"][target] = sorted(vars_state["list"][source])
        code_lines.append(f"{target} = sorted({source})")
        ops_used.append("SORT")

    def op_append():
        target = pick("list")
        token, value = pick_raw_value()
        if not target:
            target = new_name()
            vars_state["list"][target] = [value]
            code_lines.append(f"{target} = [{token}]")
        else:
            vars_state["list"][target].append(value)
            code_lines.append(f"{target}.append({token})")
        ops_used.append("APPEND")

    def op_add():
        name = pick("raw")
        addend = rand_int()
        if not name:
            name = new_name()
            base = rand_int()
            vars_state["raw"][name] = base
            code_lines.append(f"{name} = {base}")
        vars_state["raw"][name] = vars_state["raw"].get(name, 0) + addend
        code_lines.append(f"{name} += {addend}")
        ops_used.append("ADD")

    def op_assign():
        name = pick("raw") or new_name()
        value = rand_int()
        vars_state["raw"][name] = value
        code_lines.append(f"{name} = {value}")
        ops_used.append("ASSIGN")

    def op_reverse():
        name = pick("list")
        if not name:
            return
        vars_state["list"][name].reverse()
        code_lines.append(f"{name}.reverse()")
        ops_used.append("REVERSE")

    def op_pop():
        name = pick("list")
        if not name or not vars_state["list"][name]:
            return
        vars_state["list"][name].pop()
        code_lines.append(f"{name}.pop()")
        ops_used.append("POP")

    def op_zip():
        a = pick("list")
        b = pick("list")
        if not a or not b:
            return
        name = new_name()
        merged = [x for pair in zip(vars_state["list"][a], vars_state["list"][b]) for x in pair]
        vars_state["list"][name] = merged
        code_lines.append(f"{name} = [x for z in zip({a}, {b}) for x in z]")
        ops_used.append("ZIP")

    def op_extend():
        dest = pick("list")
        src = pick("list")
        src_expr = f"{src}[:]" if src else "[]"
        src_val = list(vars_state["list"][src]) if src else []
        if not dest:
            dest = new_name()
            vars_state["list"][dest] = src_val
            code_lines.append(f"{dest} = {src_expr}")
        else:
            vars_state["list"][dest].extend(src_val)
            code_lines.append(f"{dest}.extend({src_expr})")
        ops_used.append("EXTEND")

    def op_insert():
        name = pick("list")
        value = rand_int()
        if not name:
            name = new_name()
            vars_state["list"][name] = [value]
            code_lines.append(f"{name} = [{value}]")
        else:
            idx = rng.randint(0, len(vars_state["list"][name])) if vars_state["list"][name] else 0
            vars_state["list"][name].insert(idx, value)
            code_lines.append(f"{name}.insert({idx}, {value})")
        ops_used.append("INSERT")

    def op_min():
        target = pick("list")
        if not target or not vars_state["list"][target]:
            return
        name = new_name()
        vars_state["raw"][name] = min(vars_state["list"][target])
        code_lines.append(f"{name} = min({target})")
        ops_used.append("MIN")

    def op_max():
        target = pick("list")
        if not target or not vars_state["list"][target]:
            return
        name = new_name()
        vars_state["raw"][name] = max(vars_state["list"][target])
        code_lines.append(f"{name} = max({target})")
        ops_used.append("MAX")

    def op_noop():
        ops_used.append("NOOP")

    handlers = {
        "PRINT": op_print,
        "SORT": op_sort,
        "APPEND": op_append,
        "ADD": op_add,
        "ASSIGN": op_assign,
        "NOOP": op_noop,
        "REVERSE": op_reverse,
        "POP": op_pop,
        "ZIP": op_zip,
        "EXTEND": op_extend,
        "INSERT": op_insert,
        "MIN": op_min,
        "MAX": op_max,
    }

    # Bootstrap with one scalar and one list so the program is non-empty
    initial_scalar = new_name()
    scalar_value = rand_int()
    vars_state["raw"][initial_scalar] = scalar_value
    code_lines.append(f"{initial_scalar} = {scalar_value}")
    ops_used.append("ASSIGN")

    initial_list = new_name()
    list_value = rand_int()
    vars_state["list"][initial_list] = [list_value]
    code_lines.append(f"{initial_list} = [{list_value}]")
    ops_used.append("APPEND")

    for _ in range(op_count):
        op_name = rng.choice(tuple(allowed_ops))
        handler = handlers.get(op_name)
        if handler:
            handler()

    # Always end with a print to guarantee observable output
    op_print()
    return code_lines, ops_used


def execute_program(code_lines: Iterable[str]) -> str:
    """Run the generated program and capture stdout."""
    buffer = io.StringIO()
    program = "\n".join(code_lines)
    with contextlib.redirect_stdout(buffer):
        exec(program, {})  # noqa: S102 - controlled program
    return buffer.getvalue()


def format_prompt(code_lines: Iterable[str]) -> str:
    """Render the user-facing prompt."""
    body = "\n".join(f">>> {line}" for line in code_lines)
    return f"{PROMPT_PREAMBLE}{body}"


def extract_answer(text: str | None) -> str | None:
    """Pull the content inside <ANSWER> tags."""
    if not text:
        return None
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


class Actor:
    """Generate Python challenges and score model outputs."""

    def __init__(self):
        self.api_key = os.getenv("CHUTES_API_KEY")

    async def _chat(
        self,
        prompt: str,
        model: str,
        base_url: str,
        timeout: int,
        temperature: float,
        api_key: str | None,
    ) -> str:
        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key or self.api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content")
        return content

    async def evaluate(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 120,
        temperature: float = 0.0,
        api_key: str | None = None,
        task_id: int | None = None,
        seed: int | None = None,
        op_count: int = 8,
        allowed_ops: Sequence[str] | None = None,
        max_digits: int = 4,
        answer: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a Python code challenge and score the model output.

        Parameters mirror the other affinetes environments so callers
        can pass `model`, `base_url`, and `CHUTES_API_KEY`.
        """
        start = time.time()
        allowed = tuple(allowed_ops) if allowed_ops else DEFAULT_ALLOWED_OPS
        seed_value = seed if seed is not None else (task_id if task_id is not None else random.getrandbits(32))
        api_key_value = api_key or self.api_key

        code_lines, ops_used = build_program(
            seed=seed_value,
            op_count=op_count,
            allowed_ops=allowed,
            max_digits=max_digits,
        )
        prompt = format_prompt(code_lines)

        try:
            expected_output = execute_program(code_lines)
        except Exception as exc:  # pragma: no cover - generation failure is surfaced to caller
            return {
                "task_name": "python-env",
                "success": False,
                "score": 0.0,
                "error": f"Generated program failed: {exc}",
                "prompt": prompt,
                "seed": seed_value,
            }

        # Either call the model or accept a direct answer (useful for tests)
        if answer is None:
            if not api_key_value:
                return {
                    "task_name": "python-env",
                    "success": False,
                    "score": 0.0,
                    "error": "Provide CHUTES_API_KEY or pass `answer` to skip LLM call.",
                    "prompt": prompt,
                    "seed": seed_value,
                }
            model_response = await self._chat(
                prompt=prompt,
                model=model,
                base_url=base_url,
                timeout=timeout,
                temperature=temperature,
                api_key=api_key_value,
            )
        else:
            model_response = answer

        provided_output = extract_answer(model_response)

        # Accept trailing newline differences but keep formatting strict otherwise
        expected_canonical = expected_output.rstrip("\n")
        provided_canonical = provided_output.rstrip("\n") if provided_output is not None else None
        success = provided_canonical is not None and provided_canonical == expected_canonical
        score = 1.0 if success else 0.0

        return {
            "task_name": "python-env",
            "success": success,
            "score": score,
            "expected_output": expected_output,
            "provided_output": provided_output,
            "model_response": model_response,
            "prompt": prompt,
            "seed": seed_value,
            "ops_used": ops_used,
            "time_taken": time.time() - start,
        }
