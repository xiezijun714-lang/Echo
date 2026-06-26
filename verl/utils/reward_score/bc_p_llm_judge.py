import json
import logging
import os
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

class LLMJudgeContentBlockedError(RuntimeError):
    pass


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else repr(exc)


def _extract_question(extra_info: dict[str, Any] | None) -> str:
    if not isinstance(extra_info, dict):
        return ""
    for key in ("query", "question"):
        value = extra_info.get(key)
        if value:
            return str(value)
    prompt = extra_info.get("prompt") or extra_info.get("raw_prompt")
    if prompt is not None:
        try:
            messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = str(msg.get("content", ""))
                        match = re.search(r"Question:\s*(.*?)(?:\n\nFollow this structured protocol|\Z)", content, re.DOTALL)
                        return match.group(1).strip() if match else content.strip()
        except Exception:
            pass
    return ""


def _extract_final_response(solution_str: str) -> str:
    """Extract the final submitted response from a training trajectory.

    The official BrowseComp-Plus evaluator judges only the final output_text in
    the run JSON. In verl validation/training, solution_str is the full
    interaction trajectory, so we reconstruct the same final-response surface.
    """
    # Pattern 1: finish tool call in <tool_call> tags (primary)
    tool_calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", solution_str, flags=re.DOTALL)
    for tool_call in reversed(tool_calls):  # last finish call takes priority
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
        if not isinstance(arguments, dict):
            continue
        answer = arguments.get("answer", "")
        explanation = arguments.get("explanation", "")
        confidence = arguments.get("confidence", "")
        parts = []
        if explanation:
            parts.append(f"Explanation: {explanation}")
        if answer:
            parts.append(f"Exact Answer: {answer}")
        if confidence:
            parts.append(f"Confidence: {confidence}")
        if parts:
            return "\n".join(parts)

    # Pattern 2: final report format. Preserve the report body, not just the
    # answer line, because the official judge sees the complete final output.
    final_report = re.search(
        r"(?:Explanation\s*:.*?Exact Answer\s*:.*?(?:Confidence\s*:.*?)(?=\n\s*(?:<tool_call>|user\n|assistant\n)|\Z))",
        solution_str,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if final_report:
        return final_report.group(0).strip()

    exact_matches = re.findall(r"Exact Answer\s*:\s*(.+)", solution_str)
    if exact_matches:
        return f"Exact Answer: {exact_matches[-1].strip().splitlines()[0].strip()}"

    # Pattern 3: "Answer submitted:" format (FinishTool response)
    matches = re.findall(r"Answer submitted:\s*(.+)", solution_str)
    if matches:
        submitted = matches[-1].strip()
        answer = submitted.split("|", 1)[0].strip()
        return f"Exact Answer: {answer}" if answer else ""

    return ""


def _looks_like_content_block(status: int, body: str) -> bool:
    if status != 400:
        return False
    body_lower = body.lower()
    block_markers = (
        "data_inspection",
        "content filter",
        "content_filter",
        "moderation",
        "sensitive",
        "inappropriate",
        "risk",
        "security",
        "审核",
        "拦截",
        "敏感",
    )
    return any(marker in body_lower for marker in block_markers)


def _parse_judge_response(judge_response: str) -> dict[str, Any]:
    result = {
        "extracted_final_answer": None,
        "judge_reasoning": None,
        "judge_correct": False,
        "judge_confidence": None,
        "judge_parse_error": False,
    }
    if not judge_response:
        result["judge_parse_error"] = True
        return result

    # Try JSON parse first (DeepSeek often returns JSON format)
    try:
        data = json.loads(judge_response.strip())
        if isinstance(data, dict):
            result["extracted_final_answer"] = data.get("extracted_final_answer")
            result["judge_reasoning"] = data.get("reasoning")
            correct_val = str(data.get("correct", "")).strip().lower()
            if correct_val in ("yes", "no"):
                result["judge_correct"] = correct_val == "yes"
            else:
                result["judge_parse_error"] = True
            conf_raw = data.get("confidence", "")
            conf_match = re.search(r"(\d+(?:\.\d+)?)", str(conf_raw))
            if conf_match:
                result["judge_confidence"] = min(float(conf_match.group(1)), 100.0)
            return result
    except (ValueError, TypeError):
        pass

    # Fallback: plain text parsing, aligned with official evaluate_run.py.
    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["judge_reasoning"] = reasoning_match.group(1).strip()

    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"correct:\s*\"?(yes|no)\"?\b", judge_response, re.IGNORECASE)
    if correct_match:
        result["judge_correct"] = correct_match.group(1).lower() == "yes"
    else:
        result["judge_parse_error"] = True

    confidence_match = re.search(r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"confidence:\s*\"?(\d+(?:\.\d+)?)\s*%?\"?", judge_response, re.IGNORECASE)
    if confidence_match:
        confidence = float(confidence_match.group(1))
        result["judge_confidence"] = min(confidence, 100.0)

    return result


async def _post_chat_completion(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    max_retries: int = 8,
) -> dict[str, Any]:
    last_exception = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        message = body[:1000]
                        if _looks_like_content_block(resp.status, body):
                            raise LLMJudgeContentBlockedError(f"LLM judge content blocked HTTP {resp.status}: {message}")
                        if 400 <= resp.status < 500 and resp.status != 429:
                            logger.error("LLM judge request failed with client error HTTP %s: %s", resp.status, message)
                            raise RuntimeError(f"LLM judge HTTP {resp.status}: {message}") from None
                        last_exception = RuntimeError(f"LLM judge HTTP {resp.status}: {message}")
                    else:
                        return json.loads(body)
        except LLMJudgeContentBlockedError:
            raise
        except Exception as exc:
            last_exception = RuntimeError(_exception_message(exc))
        if attempt < max_retries - 1:
            import asyncio

            await asyncio.sleep(min(2**attempt, 30))
    if last_exception:
        raise last_exception
    raise RuntimeError(f"LLM judge request failed: {url}")


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer=None,
    judge_max_tokens: int = 4096,
    judge_temperature: float = 0.0,
    judge_top_p: float = 1.0,
    **kwargs,
):
    judge_model = os.getenv("BCP_JUDGE_MODEL", "Deepseek-V4-Flash")
    judge_api_base = os.getenv("BCP_JUDGE_API_BASE", "http://localhost:8000/v1")
    judge_api_key_env = os.getenv("BCP_JUDGE_API_KEY_ENV", "ONEAPI_KEY")
    judge_max_tokens = int(os.getenv("BCP_JUDGE_MAX_TOKENS", str(judge_max_tokens)))
    judge_temperature = float(os.getenv("BCP_JUDGE_TEMPERATURE", str(judge_temperature)))
    judge_top_p = float(os.getenv("BCP_JUDGE_TOP_P", str(judge_top_p)))

    question = _extract_question(extra_info)
    response = _extract_final_response(solution_str)
    if not response:
        return {
            "score": 0.0,
            "acc": 0.0,
            "judge_correct": 0.0,
            "judge_parse_error": 1.0,
            "judge_confidence": -1.0,
        }
    if not question:
        logger.warning("BC-P LLM judge received empty question; falling back to judging with an empty [question] field.")
    judge_prompt = GRADER_TEMPLATE.format(question=question, response=response, correct_answer=str(ground_truth))
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": judge_temperature,
        "top_p": judge_top_p,
        "max_tokens": judge_max_tokens,
    }
    judge_top_k = os.getenv("BCP_JUDGE_TOP_K")
    if judge_top_k:
        payload["top_k"] = int(judge_top_k)
    api_key = os.getenv(judge_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"BC-P LLM judge API key not set. Export the key via the '{judge_api_key_env}' "
            "environment variable (configurable through BCP_JUDGE_API_KEY_ENV)."
        )
    url = f"{judge_api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        outputs = await _post_chat_completion(url, payload, headers=headers)
    except LLMJudgeContentBlockedError as exc:
        logger.warning("BC-P LLM judge content blocked; assigning zero reward: %s", exc)
        return {
            "score": 0.0,
            "acc": 0.0,
            "judge_correct": 0.0,
            "judge_parse_error": 1.0,
            "judge_confidence": -1.0,
        }
    except Exception as exc:
        logger.warning("BC-P LLM judge request failed; assigning zero reward: %s", _exception_message(exc))
        return {
            "score": 0.0,
            "acc": 0.0,
            "judge_correct": 0.0,
            "judge_parse_error": 1.0,
            "judge_confidence": -1.0,
        }
    judge_response = outputs["choices"][0]["message"]["content"]
    parsed = _parse_judge_response(judge_response)
    score = 1.0 if parsed["judge_correct"] else 0.0
    return {
        "score": score,
        "acc": score,
        "judge_correct": 1.0 if parsed["judge_correct"] else 0.0,
        "judge_parse_error": 1.0 if parsed["judge_parse_error"] else 0.0,
        "judge_confidence": parsed["judge_confidence"] if parsed["judge_confidence"] is not None else -1.0,
    }
