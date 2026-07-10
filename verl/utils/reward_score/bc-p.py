import logging
import json
import re
import string

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip, remove articles / punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_answer(solution_str: str) -> str | None:
    """Extract answer from BC-Plus final text or finish(answer=...) calls."""
    pattern = r"Exact Answer\s*:\s*(.+)"
    matches = re.findall(pattern, solution_str, re.DOTALL)
    if matches:
        # Take the first match; strip trailing whitespace/newlines
        answer = matches[0].strip()
        # Truncate at newline to avoid capturing subsequent sections
        answer = answer.split("\n")[0].strip()
        return answer

    # Paper-style tool submission: <tool_call>{"name": "finish", "arguments": ...}</tool_call>
    tool_calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", solution_str, flags=re.DOTALL)
    for tool_call in tool_calls:
        try:
            payload = json.loads(tool_call)
        except Exception:
            continue
        name = payload.get("name") or payload.get("function", {}).get("name")
        if str(name).lower() != "finish":
            continue
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        answer = arguments.get("answer") if isinstance(arguments, dict) else None
        if answer:
            return str(answer).strip()

    # Fallback for the tool response emitted by FinishTool.
    matches = re.findall(r"Answer submitted:\s*(.+)", solution_str)
    if matches:
        return matches[0].split("|")[0].strip()
    return None


def _count_tool_calls(solution_str: str) -> dict:
    """Count search tool calls and extract search queries."""
    n_search = len(re.findall(r'"name"\s*:\s*"search"', solution_str))

    # Extract search queries from query parameter (single string)
    queries = re.findall(r'"query"\s*:\s*"([^"]+)"', solution_str)

    return {
        "n_search": n_search,
        "queries": queries,
    }


def _process_reward(solution_str: str) -> float:
    """Compute process reward based on behavioral signals.

    Rewards:
        +0.2  produced Exact Answer format (anti token-waste / structured output)
        +0.1  no repeated search queries (anti death-loop)
        +0.1  called search at least once (anti direct-guess)
        +0.1  called search at least 2 times with distinct queries (encourage exploration)
        Max process total: 0.5  (must stay below outcome reward 1.0)
    """
    reward = 0.0
    tools = _count_tool_calls(solution_str)

    # Produced Exact Answer: +0.2
    if _extract_answer(solution_str) is not None:
        reward += 0.2

    # Called search at least once: +0.1
    if tools["n_search"] > 0:
        reward += 0.1

    # No repeated queries: +0.1
    queries = tools["queries"]
    if queries and len(queries) == len(set(queries)):
        reward += 0.1

    # At least 2 distinct search queries: +0.1
    if queries and len(set(queries)) >= 2:
        reward += 0.1

    return reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Outcome reward (binary, 0 or 1.0)
    outcome = 0.0
    predicted = _extract_answer(solution_str)
    if predicted:
        expected = str(ground_truth).strip()
        if predicted.lower() == expected.lower():
            outcome = 1.0
        elif _normalize(predicted) == _normalize(expected):
            outcome = 1.0
        else:
            norm_pred = _normalize(predicted)
            norm_exp = _normalize(expected)
            if norm_exp and norm_pred:
                if norm_exp in norm_pred or norm_pred in norm_exp:
                    outcome = 1.0

    return {
        "score": outcome,   # pure outcome reward
        "acc": outcome,    # pure accuracy (0 or 1) for validation
    }
