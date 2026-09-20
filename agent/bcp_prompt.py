import argparse
from pathlib import Path
import pandas as pd


PAPER_SYSTEM_MARKER = "You have access to the following functions:"

PAPER_SYSTEM_PROMPT = """You are a meticulous and strategic research agent. Your primary function is to conduct comprehensive, multi-step research to deliver a thorough, accurate, and well-supported report in response to the user's query. Your operation is guided by these core principles:
- Rigor: Execute every step of the research process with precision and attention to detail.
- Objectivity: Synthesize information based on the evidence gathered, not on prior assumptions. Note and investigate conflicting information.
- Thoroughness: Never settle for a surface-level answer. Always strive to uncover the underlying details, context, and data.
- Transparency: Your reasoning process should be clear at every step, linking evidence from your research directly to your conclusions.

You have access to the following functions:
---- BEGIN FUNCTION #1: search ----
Description: Performs a web search: supply a string 'query' and optional 'topk'. The tool retrieves the top 'topk' results (default 10) for the query, returning their docid, url, and document content (may be truncated based on token limits).
Parameters:
(1) query (string, required): The query string for the search.
(2) topk (integer, optional): Return the top k pages.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: open_page ----
Description: Open a page by docid or URL and return the complete content. Provide either 'docid' or 'url'; if both are provided, prefer 'docid'. The docid or URL must come from prior search tool results.
Parameters:
(1) docid (string, optional): Document ID from search results to resolve and fetch.
(2) url (string, optional): Absolute URL from search results to fetch.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: finish ----
Description: Return the final result when you have a definitive answer or cannot progress further. Provide a concise answer plus a brief, evidence-grounded explanation.
Parameters:
(1) answer (string, required): A succinct, final answer.
(2) explanation (string, required): A brief explanation for your final answer. For this section only, cite evidence documents inline by placing their docids in square brackets at the end of sentences (e.g., [20]). Do not include citations anywhere else.
(3) confidence (string, optional): Confidence: your confidence score between 0% and 100% for your answer.
---- END FUNCTION #3 ----

If you choose to call a function only reply in the following format with no suffix:
< function = example_function_name >
< parameter = example_parameter_1 > value_1 </ parameter >
< parameter = example_parameter_2 >
This is the value for the second parameter that can span multiple lines
</ parameter >
</ function >

Reminder:
Function calls must follow the specified format, start with <function=function_name> and end with </function=function_name>. Required parameters must be specified. You may provide optional reasoning for your function call in natural language before the function call, but not after. If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
"""


def _extract_query_from_prompt(prompt) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    for message in prompt:
        if dict(message).get("role") != "user":
            continue
        content = str(dict(message).get("content") or "")
        marker = "Question:"
        if marker in content:
            content = content.split(marker, 1)[1].strip()
            return content.split("\n\nYour response should be", 1)[0].strip()
        return content.strip()
    return ""


def _tool_usage_block(tool_style: str) -> str:
    if tool_style in {"paper_fc", "seed_oss"}:
        return """You can search one query:
< function = search >
< parameter = query > Query </ parameter >
< parameter = topk >10 </ parameter >
</ function >

Or you can search multiple queries in one turn:
< function = search >
< parameter = query > Query1 </ parameter >
< parameter = topk >5 </ parameter >
</ function >
< function = search >
< parameter = query > Query2 </ parameter >
< parameter = topk >5 </ parameter >
</ function >

Use open_page to fetch a web page:
< function = open_page >
< parameter = docid > docid </ parameter >
</ function >
or
< function = open_page >
< parameter = url > url </ parameter >
</ function >"""

    if tool_style != "hermes":
        raise ValueError(f"Unsupported tool_style: {tool_style}")
    return """You can search one query:
<tool_call>
{"name": "search", "arguments": {"query": "Query", "topk": 10}}
</tool_call>

Or you can search multiple queries in one turn:
<tool_call>
{"name": "search", "arguments": {"query": "Query1", "topk": 5}}
</tool_call>
<tool_call>
{"name": "search", "arguments": {"query": "Query2", "topk": 5}}
</tool_call>

Use open_page to fetch a source document:
<tool_call>
{"name": "open_page", "arguments": {"docid": "docid"}}
</tool_call>

Use finish to submit your final answer:
<tool_call>
{"name": "finish", "arguments": {"answer": "succinct final answer", "explanation": "evidence-backed explanation with docid citations", "confidence": "80%"}}
</tool_call>"""


def build_user_prompt(query: str, tool_style: str = "hermes") -> str:
    tool_usage = _tool_usage_block(tool_style)
    return f"""You need to answer the given question by interacting with a search engine, using the search and open tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and open tools multiple times. Question:
{query}

Follow this structured protocol for to find the answer:

Phase 1: Deconstruction & Strategy
1. Deconstruct the Query:
- Analyze the user's prompt to identify the core question(s).
- Isolate key entities, concepts, and the relationships between them.
- Explicitly list all constraints, conditions, and required data points (e.g., dates, quantities, specific names).
2. Hypothesize & Brainstorm:
- Based on your knowledge, brainstorm potential search vectors, keywords, synonyms, and related topics that could yield relevant information.
- Consider multiple angles of inquiry to approach the problem.
3. Verification Checklist:
- Create a Verification Checklist based on the query's constraints and required data points. This checklist will be your guide throughout the process and used for final verification.

Phase 2: Iterative Research & Discovery
1. Tools:
- search: Use for broad discovery of sources and to get initial snippets.
- open_page: Mandatory follow-up for any promising search result. Snippets are insufficient; you must analyze the full context of the source document.
2. Query Strategy:
- Start with moderately broad queries to map the information landscape.
- Narrow your focus as you learn more.
- Do not repeat the exact same query. If a query fails, rephrase it or change your angle of attack.
- Execute a minimum of 5 tool calls for simple queries and up to 50 tool calls for complex ones. Do not terminate prematurely.
- Never simulate tool call output.

Phase 3: Synthesis & Analysis
1. Continuous Synthesis: Throughout the research process, continuously integrate new information with existing knowledge. Build a coherent narrative and understanding of the topic.
2. Triangulate Critical Data: For any crucial fact, number, date, or claim, you must seek to verify it across at least two independent, reliable sources. Note any discrepancies.
3. Handle Dead Ends: If you are blocked, do not give up. Broaden your search scope, try alternative keywords, or research related contextual information to uncover new leads. Assume a discoverable answer exists and exhaust all reasonable avenues.
4. Maintain a "Fact Sheet": Internally, keep a running list of key facts, figures, dates, and their supporting sources. This will be crucial for the final report.

Phase 4: Verification & Final Report Formulation
1. Systematic Verification: Before writing the final answer, halt your research and review your Verification Checklist created in Phase 1. For each item on the checklist, confirm you have sufficient, well-supported evidence from the documents you have opened.
2. Mandatory Re-research: If any checklist item is unconfirmed or the evidence is weak, it is mandatory to return to Phase 2 to conduct further targeted research. Do not formulate an answer based on incomplete information.
3. Never give up, no matter how complex the query, you will not give up until you find the corresponding information.
4. Construct the Final Report:
- Once all checklist items are confidently verified, synthesize all gathered facts into a comprehensive and well-structured answer.
- Directly answer the user's original query.
- Ensure all claims, numbers, and key pieces of information in your report are clearly supported by the research you conducted.

Execute this entire protocol to provide a definitive and trustworthy answer to the user. You can search one queries:

{tool_usage}

Your final answer should contain:
1. Explanation: your explanation for your final answer. For this explanation section only, cite evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].
2. Exact Answer: your succinct, final answer.
3. Confidence: your confidence score between 0% and 100% for your answer.
Use finish tool to submit your answer."""


def build_prompt(row: pd.Series, tool_style: str = "hermes") -> list[dict[str, str]]:
    query = str(row.get("query") or "").strip()
    if not query:
        query = _extract_query_from_prompt(row.get("prompt"))
    if not query:
        raise ValueError("Could not recover query for row")
    return [
        {"role": "system", "content": PAPER_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, tool_style=tool_style)},
    ]


def patch_file(input_path: Path, output_path: Path, tool_style: str) -> None:
    df = pd.read_parquet(input_path)
    if "prompt" not in df.columns:
        raise ValueError(f"{input_path} does not contain a prompt column")
    df = df.copy()
    df["prompt"] = df.apply(lambda row: build_prompt(row, tool_style=tool_style), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"wrote {output_path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-name", default="train.parquet")
    parser.add_argument("--test-name", default="test.parquet")
    parser.add_argument("--train-out", default="train.paper.parquet")
    parser.add_argument("--test-out", default="test.paper.parquet")
    parser.add_argument("--tool-style", choices=["hermes", "paper_fc", "seed_oss"], default="hermes")
    args = parser.parse_args()

    patch_file(args.data_dir / args.train_name, args.data_dir / args.train_out, args.tool_style)
    patch_file(args.data_dir / args.test_name, args.data_dir / args.test_out, args.tool_style)


if __name__ == "__main__":
    main()
