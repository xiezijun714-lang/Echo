"""OpenPage tool — retrieve a document by docid with a fixed truncation budget."""

import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import requests

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_RETRIES = 2
INITIAL_RETRY_DELAY = 1
# HTTP status codes that are NOT transient — do not retry
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}


class OpenPageTool(BaseTool):
    """Fetch a document by docid and return its beginning up to max_chars."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.max_chars = config.get("max_chars", 16000)
        self.timeout = config.get("timeout", 30)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        docid = parameters.get("docid", "")

        # Sanitize: ensure docid is a string.
        if not isinstance(docid, str):
            docid = str(docid)

        # Strip common copy-paste artifacts like "[docid: 12345]"
        docid = docid.strip().strip("[]")
        if docid.startswith("docid:"):
            docid = docid[len("docid:"):].strip()

        if not docid:
            err = json.dumps({"error": "Missing 'docid' parameter."})
            return ToolResponse(text=err), 0.0, {}

        url = self.retrieval_service_url
        payload = {"docid": docid}
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)

                # Non-retryable errors (4xx client errors) — return immediately
                if resp.status_code in NON_RETRYABLE_STATUS:
                    try:
                        detail = resp.json()
                        msg = detail.get("detail", detail.get("error", resp.text[:200]))
                    except Exception:
                        msg = resp.text[:200]
                    err = json.dumps({"error": f"Document retrieval failed (HTTP {resp.status_code}): {msg}. Do NOT retry with the same docid. Try a different docid or search query."})
                    return ToolResponse(text=err), 0.0, {}

                resp.raise_for_status()
                data = resp.json()

                if data.get("error"):
                    # Server returned a semantic error (e.g., docid not found)
                    return ToolResponse(text=json.dumps({"error": f"{data['error']}. Try a different docid from search results."})), 0.0, {}

                document = data.get("document") or {}
                content = str(document.get("contents") or "")
                original_chars = len(content)
                if self.max_chars and original_chars > self.max_chars:
                    content = content[: self.max_chars] + "\n...(truncated)"
                result_text = (
                    f"Document [docid: {docid}] "
                    f"(showing {len(content)}/{original_chars} chars):\n{content}"
                )
                return ToolResponse(text=json.dumps({"result": result_text}, ensure_ascii=False)), 0.0, {}

            except requests.exceptions.ConnectionError:
                last_error = "Connection refused — retrieval server may be down"
                logger.warning(f"open_page connection error for docid={docid}, attempt {attempt+1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.timeout}s"
                logger.warning(f"open_page timeout for docid={docid}, attempt {attempt+1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))

        err = json.dumps({"error": f"Failed to fetch document {docid} after {MAX_RETRIES} attempts: {last_error}. Try a different docid or search query."})
        return ToolResponse(text=err), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs):
        return 0.0

    async def release(self, instance_id: str, **kwargs):
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
