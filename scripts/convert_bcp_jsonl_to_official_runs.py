#!/usr/bin/env python3
"""Convert VERL BrowseComp-Plus validation jsonl dumps to official run JSONs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


QUESTION_RE = re.compile(
    r"(?:^|\n)Question:\n(?P<question>.*?)(?:\n\nFollow this structured protocol|\n\nYou can search|\Z)",
    re.DOTALL,
)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_SUBMITTED_RE = re.compile(
    r"Answer submitted:\s*(?P<answer>.*?)(?:\s*\|\s*Explanation:\s*(?P<explanation>.*?))?(?:\s*\|\s*Confidence:\s*(?P<confidence>.*?))?(?:\s*$|\n)",
    re.DOTALL,
)
FINAL_REPORT_RE = re.compile(
    r"(Explanation\s*:.*?Exact Answer\s*:.*?(?:Confidence\s*:.*?)(?=\s*$|\n\s*(?:user|assistant|system)\n|<tool_call>|<tool_response>))",
    re.IGNORECASE | re.DOTALL,
)
DOCID_RE = re.compile(r"\bdocid:\s*([^\]\s,}]+)|Doc\s+\d+\s+\([^)]*\)\s+\[docid:\s*([^\]]+)\]", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_ground_truth(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    by_query: dict[str, dict[str, str]] = {}
    by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            query_id = str(obj["query_id"])
            query = normalize_text(str(obj["query"]))
            by_query[query] = {
                "query_id": query_id,
                "question": obj["query"],
                "answer": obj["answer"],
            }
            by_id[query_id] = obj["answer"]
    return by_query, by_id


def extract_question(record: dict[str, Any]) -> str:
    for key in ("query", "question"):
        if key in record and record[key]:
            return normalize_text(str(record[key]))

    text = str(record.get("input", ""))
    match = QUESTION_RE.search(text)
    if match:
        return normalize_text(match.group("question"))
    return ""


def iter_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        raw = match.group(1)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            calls.append(obj)
    return calls


def format_finish_response(arguments: dict[str, Any]) -> str:
    answer = str(arguments.get("answer", "")).strip()
    explanation = str(arguments.get("explanation", "")).strip()
    confidence = str(arguments.get("confidence", "")).strip()

    parts = []
    if explanation:
        parts.append(f"Explanation: {explanation}")
    if answer:
        parts.append(f"Exact Answer: {answer}")
    if confidence:
        parts.append(f"Confidence: {confidence}")
    return "\n".join(parts)


def extract_final_response(output: str) -> tuple[str, bool]:
    calls = iter_tool_calls(output)
    for call in reversed(calls):
        if call.get("name") != "finish":
            continue
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if isinstance(arguments, dict):
            response = format_finish_response(arguments).strip()
            if response:
                return response, True

    submitted_matches = list(ANSWER_SUBMITTED_RE.finditer(output))
    if submitted_matches:
        match = submitted_matches[-1]
        answer = (match.group("answer") or "").strip()
        explanation = (match.group("explanation") or "").strip()
        confidence = (match.group("confidence") or "").strip()
        response = format_finish_response(
            {"answer": answer, "explanation": explanation, "confidence": confidence}
        ).strip()
        if response:
            return response, True

    report_matches = list(FINAL_REPORT_RE.finditer(output))
    if report_matches:
        return report_matches[-1].group(1).strip(), True

    return "", False


def collect_tool_stats(output: str) -> tuple[dict[str, int], list[str]]:
    counts = Counter()
    for call in iter_tool_calls(output):
        name = call.get("name")
        if isinstance(name, str):
            counts[name] += 1

    docids = []
    seen = set()
    for match in DOCID_RE.finditer(output):
        docid = (match.group(1) or match.group(2) or "").strip()
        if docid and docid not in seen:
            seen.add(docid)
            docids.append(docid)
    return dict(counts), docids


def convert_file(input_jsonl: Path, output_dir: Path, gt_by_query: dict[str, dict[str, str]]) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    with input_jsonl.open("r", encoding="utf-8") as f:
        for source_index, line in enumerate(f):
            record = json.loads(line)
            question = extract_question(record)
            gt = gt_by_query.get(question)
            if gt is None:
                stats["unmatched"] += 1
                query_id = record.get("query_id") or record.get("uid") or f"unmatched_{source_index}"
                correct_answer = record.get("gts", "")
            else:
                stats["matched"] += 1
                query_id = gt["query_id"]
                correct_answer = gt["answer"]

            output = str(record.get("output", ""))
            final_response, completed = extract_final_response(output)
            tool_counts, retrieved_docids = collect_tool_stats(output)

            status = "completed" if completed else "incomplete"
            result = [{"type": "output_text", "output": final_response}] if completed else []
            if completed:
                stats["completed"] += 1
            else:
                stats["incomplete"] += 1

            run_obj = {
                "query_id": str(query_id),
                "status": status,
                "result": result,
                "tool_call_counts": tool_counts,
                "retrieved_docids": retrieved_docids,
                "metadata": {
                    "source_jsonl": str(input_jsonl.resolve()),
                    "source_index": source_index,
                    "uid": record.get("uid"),
                    "model": input_jsonl.stem,
                    "old_acc": record.get("acc"),
                    "old_score": record.get("score"),
                    "old_judge_correct": record.get("judge_correct"),
                    "old_judge_parse_error": record.get("judge_parse_error"),
                    "ground_truth": correct_answer,
                },
            }

            out_path = output_dir / f"{query_id}.json"
            with out_path.open("w", encoding="utf-8") as out:
                json.dump(run_obj, out, indent=2, ensure_ascii=False)

    stats["written"] = stats["matched"] + stats["unmatched"]
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("jsonl", type=Path, nargs="+")
    args = parser.parse_args()

    gt_by_query, _ = load_ground_truth(args.ground_truth)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for input_jsonl in args.jsonl:
        step = input_jsonl.stem
        out_dir = args.output_root / f"val{step}"
        stats = convert_file(input_jsonl, out_dir, gt_by_query)
        all_stats[str(input_jsonl)] = {"output_dir": str(out_dir), **stats}

    print(json.dumps(all_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
