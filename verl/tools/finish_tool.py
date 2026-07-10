"""
Finish tool — signals the agent has reached a final answer.

Mirrors the finish() tool in the SUPO/Fold-GRPO BrowseComp-Plus setup.
The model calls finish(answer=..., explanation=...) to submit its answer
and terminate the multi-turn interaction.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FinishTool(BaseTool):
    """Signal that the agent has a final answer and stop the interaction."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        answer = parameters.get("answer", "")
        explanation = parameters.get("explanation", "")
        confidence = parameters.get("confidence", "")

        if not answer:
            return (
                ToolResponse(text=json.dumps({"error": "Missing 'answer' parameter."})),
                0.0,
                {},
            )

        # Build confirmation response
        parts = [f"Answer submitted: {answer}"]
        if explanation:
            parts.append(f"Explanation: {explanation}")
        if confidence:
            parts.append(f"Confidence: {confidence}")

        return (
            ToolResponse(text=json.dumps({"result": " | ".join(parts)}, ensure_ascii=False)),
            0.0,
            {},
        )

    async def calc_reward(self, instance_id: str, **kwargs):
        return 0.0

    async def release(self, instance_id: str, **kwargs):
        pass
