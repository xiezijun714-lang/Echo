# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import requests
import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SEMANTIC_SELECTION_TOPK = 5
SEMANTIC_SELECTION_RECENT_TURNS = 3
SEMANTIC_SELECTION_QUERY_UNITS = 3
SEMANTIC_SELECTION_HINT_CHARS = 1200
SEMANTIC_SELECTION_EMBED_TIMEOUT = 300
TOOL_CALL_MARKER_PATTERN = r'<\|FunctionCallBegin\||<tool_call>|<function_call>'
SEMANTIC_SELECTION_INSTRUCTION = (
    "Given a task state, retrieve relevant interaction turns that help continue solving the task."
)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


@dataclass
class TrajectoryOutput:
    """A trajectory segment emitted by SUPO-style context compression."""
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    response_turn_ids: list[int]
    response_finding_turn_ids: list[int]
    response_selection_mask: list[int]
    is_final: bool = False


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}

        # State for SUPO-style context compression.
        self.original_messages: list[dict] = []
        self.original_prompt_ids: list[int] = []
        self.summary_count: int = 0
        self.trajectory_outputs: list[TrajectoryOutput] = []
        self.is_summarizing: bool = False
        self.overlong: bool = False
        # Response tokens accumulated for the current trajectory segment.
        self.current_traj_prompt_ids: list[int] = []
        self.accumulated_response_ids: list[int] = []
        self.accumulated_response_mask: list[int] = []
        self.accumulated_logprobs: list[float] = []
        self.accumulated_response_turn_ids: list[int] = []
        self.accumulated_response_finding_turn_ids: list[int] = []
        self.accumulated_response_selection_mask: list[int] = []

        # Turn-level history used by ECHO context reconstruction.
        self.turn_history: list[dict] = []
        self.current_selected_traj_indices: list[int] = []
        # The active prompt view includes automatic recent-turn retention.
        self.current_selected_turn_ids: list[int] = []
        # Only explicit selector choices create cross-segment graph edges;
        # automatically retained recent context remains a prompt-only detail.
        self.current_model_selected_turn_ids: list[int] = []
        self.memory_graph_valid: bool = True
        self.echo_graph_version: int = 2
        self.next_turn_id: int = 0
        # A turn that crossed the context boundary is still the active local
        # context. Keep it separate from turn_history until the next assistant
        # response writes its <sum_last_turn> finding.
        self.pending_turn: dict | None = None
        self.echo_pending_local_turn: dict | None = None
        self.echo_pending_local_messages: list[dict] = []
        self.echo_selection_prefix_ids: list[int] | None = None
        # Completed turns generated after the latest ECHO reconstruction. The
        # first turn depends on explicit selections; later turns form a sparse
        # local chain. gamma_turn=1 recovers segment-level credit.
        self.current_segment_turn_ids: list[int] = []
        # Auxiliary selector actions live outside the reasoning DAG. Each event
        # maps one serialized selection span to the first downstream turn.
        self.echo_selection_events: list[dict[str, Any]] = []
        self.current_echo_selection_id: int | None = None
        self.next_echo_selection_id: int = 0

        # ECHO diagnostics are carried on every emitted segment so the trainer
        # can distinguish "never reached the threshold" from a failed split.
        self.echo_trigger_count: int = 0
        self.echo_trigger_reasons: list[str] = []
        self.echo_trigger_lengths: list[int] = []
        self.echo_trigger_prompt_lengths: list[int] = []
        self.echo_trigger_response_lengths: list[int] = []
        self.echo_trigger_tool_lengths: list[int] = []
        self.echo_trigger_history_lengths: list[int] = []
        self.echo_selection_count: int = 0
        self.echo_selection_parse_failures: int = 0
        self.echo_selection_response_lengths: list[int] = []
        self.echo_selection_stop_reasons: list[str] = []
        self.echo_tool_parse_failures: int = 0
        self.echo_missing_finding_count: int = 0
        self.echo_finish_before_threshold_count: int = 0
        self.echo_reconstruction_prompt_lengths: list[int] = []
        self.echo_prompt_budget_overflow_count: int = 0


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.chat_template_tool_schemas = (
            self.tool_schemas if getattr(self.rollout_config.multi_turn, "inject_tool_schemas", True) else None
        )
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_model_len = getattr(self.rollout_config, "max_model_len", None)

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

        # SUPO/ECHO context compression configuration.
        self.enable_summarization = getattr(self.rollout_config.multi_turn, 'enable_summarization', False)
        self.max_summary_rounds = getattr(self.rollout_config.multi_turn, 'max_summary_rounds', 2)
        self.train_summary_tokens = getattr(self.rollout_config.multi_turn, 'train_summary_tokens', True)
        self.working_context_length = getattr(self.rollout_config.multi_turn, 'working_context_length', 8192)
        # "summary" uses generated summaries; "truncate" drops oldest context;
        # "echo_e2e" lets the actor select turns; "semantic_selection" ranks turns by embedding similarity.
        self.context_compression_method = getattr(self.rollout_config.multi_turn, 'context_compression_method', 'summary')
        self.truncate_keep_ratio = float(getattr(self.rollout_config.multi_turn, 'truncate_keep_ratio', 0.5) or 0.5)
        if not 0.0 < self.truncate_keep_ratio <= 1.0:
            raise ValueError(f"truncate_keep_ratio must be in (0, 1], got {self.truncate_keep_ratio}")
        self.echo_recent_turns = int(getattr(self.rollout_config.multi_turn, 'echo_recent_turns', 3) or 0)
        self.echo_graph_version = int(getattr(self.rollout_config.multi_turn, 'echo_graph_version', 2) or 2)
        if self.echo_graph_version not in {1, 2}:
            raise ValueError(f"echo_graph_version must be 1 or 2, got {self.echo_graph_version}")
        self.selection_max_turns = int(getattr(self.rollout_config.multi_turn, 'selection_max_turns', 8) or 0)
        self.selection_max_new_tokens = int(
            getattr(self.rollout_config.multi_turn, 'selection_max_new_tokens', 1024) or 1024
        )
        if self.selection_max_new_tokens <= 0:
            raise ValueError(
                f"selection_max_new_tokens must be positive, got {self.selection_max_new_tokens}"
            )
        self.sum_last_turn_max_chars = int(getattr(self.rollout_config.multi_turn, 'sum_last_turn_max_chars', 300) or 300)
        self.summary_max_chars = int(getattr(self.rollout_config.multi_turn, 'summary_max_chars', 3072) or 3072)
        self.semantic_selection_full_observation = bool(
            getattr(self.rollout_config.multi_turn, 'semantic_selection_full_observation', False)
        )
        self.semantic_selection_full_observation_max_chars = int(
            getattr(self.rollout_config.multi_turn, 'semantic_selection_full_observation_max_chars', 4000) or 4000
        )
        self.semantic_embed_url = self._infer_semantic_embed_url()
        self.summary_instruction = getattr(
            self.rollout_config.multi_turn,
            'summary_instruction',
            "System:\n"
            "You are a helpful agent interacting with a function calling environment to solve the problem.\n"
            "The interaction history is now too long. Please summarize the interaction history.\n\n"
            "Rules:\n"
            "- Output exactly one <summary>...</summary> block.\n"
            "- Do not call any function/tool in this turn.\n"
            "- Do not include <think>, tool calls, markdown fences, or text outside the summary tags.\n"
            "- Keep the important information needed to continue solving the problem.\n\n"
            "<summary>\n"
            "Task Objective:\n"
            "- Original problem: [State the user problem.]\n\n"
            "Current State:\n"
            "- Key facts and constraints:\n"
            "  - [Important verified fact, constraint, or intermediate result]\n"
            "- Code or tool outputs:\n"
            "  - [Important result from previous function/tool calls]\n"
            "- Errors or failed attempts:\n"
            "  - [Known dead end, exception, or incorrect approach]\n\n"
            "Plan:\n"
            "- Next step:\n"
            "  - [Exact next action to continue solving the problem]\n"
            "</summary>"
        )
        self.selection_instruction = getattr(
            self.rollout_config.multi_turn,
            'selection_instruction',
            "System:\n"
            "Your operational context is full. Select prior interaction turns that are necessary to continue solving the task.\n\n"
            "Rules:\n"
            "- Output exactly one <selection>...</selection> block.\n"
            "- Do not call any function/tool in this turn.\n"
            "- Do not answer the original task.\n"
            "- Do not include <think>, tool calls, markdown fences, or text outside the selection tags.\n"
            "- Select only turns that contain reusable evidence, constraints, failed attempts, or the next planned action.\n"
            "- Prefer older turns that are not already covered by automatically retained recent turns.\n"
            "- If no older turn is necessary, return an empty <selection></selection> block.\n\n"
            "Valid turns:\n"
            "{turn_list}\n\n"
            "<selection>\n"
            "[zero or more lines, each formatted as turn_N: reason]\n"
            "</selection>"
        )
        self.sum_last_turn_instruction = getattr(
            self.rollout_config.multi_turn,
            'sum_last_turn_instruction',
            "# Tool Result Summary Protocol\n"
            "After receiving a tool/function response, your next assistant message must include exactly one "
            "<sum_last_turn>...</sum_last_turn> block before any new tool/function call.\n"
            "The block should be one concise factual sentence summarizing only the latest tool/function result.\n"
            "After the block, continue in the same assistant message with the next tool/function call, "
            "or call finish if the final answer is ready. Do not stop after the summary."
        )
        self.sum_last_turn_hint = getattr(
            self.rollout_config.multi_turn,
            'sum_last_turn_hint',
            "Briefly record the latest tool result in <sum_last_turn>...</sum_last_turn>, then continue with the next action."
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        messages = deepcopy(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        agent_data.echo_graph_version = self.echo_graph_version

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # SUPO can emit multiple trajectory segments for one rollout.
        if self.enable_summarization:
            if not agent_data.overlong:
                # A terminal finish can arrive immediately after a tool response,
                # leaving the newest pending ECHO turn without another processing
                # callback. Commit it before serializing the final segment.
                if self._uses_sum_last_turn_history() and agent_data.pending_turn:
                    # There is no following assistant message after a terminal
                    # tool call. Do not parse the current response here: its
                    # ``<sum_last_turn>`` belongs to the preceding tool turn.
                    self._finalize_pending_echo_turn(agent_data, "")
            if not agent_data.overlong and agent_data.accumulated_response_ids:
                self._save_final_trajectory(agent_data)

            # 生成多条AgentLoopOutput
            outputs = []
            multi_modal_data_dict = {}
            if agent_data.image_data is not None:
                multi_modal_data_dict["images"] = agent_data.image_data
            if agent_data.video_data is not None:
                multi_modal_data_dict["videos"] = agent_data.video_data

            rollout_id = f"{request_id}_rollout"
            uid = kwargs.get("uid", request_id)
            is_padding = kwargs.get("is_padding", False)  # Get is_padding marker for SUPO unpad
            reward_model = kwargs.get("reward_model", {})  # Get reward_model for SUPO (contains ground_truth)
            data_source = kwargs.get("data_source", "unknown")  # Get data_source for metrics
            final_selected_traj_indices = (
                sorted(set(agent_data.current_selected_traj_indices))
                if self._is_turn_selection_method()
                else []
            )
            final_selected_turn_ids = (
                sorted(set(agent_data.current_selected_turn_ids))
                if self._is_turn_selection_method()
                else []
            )
            memory_graph = self._build_echo_memory_graph(agent_data) if self.context_compression_method == "echo_e2e" else None
            echo_diagnostics = self._echo_diagnostics(agent_data)

            for i, traj in enumerate(agent_data.trajectory_outputs):
                # Truncate prompt and response to fit within max_model_len
                # prompt_ids: take last prompt_length tokens if too long (left truncate)
                # response_ids: take first response_length tokens if too long (right truncate)
                truncated_prompt_ids = traj.prompt_ids[-self.prompt_length:] if len(traj.prompt_ids) > self.prompt_length else traj.prompt_ids
                truncated_response_ids = traj.response_ids[:self.response_length]
                truncated_response_mask = traj.response_mask[:self.response_length]
                truncated_logprobs = traj.response_logprobs[:self.response_length] if traj.response_logprobs else None
                truncated_response_turn_ids = traj.response_turn_ids[:self.response_length]
                truncated_response_finding_turn_ids = traj.response_finding_turn_ids[:self.response_length]
                truncated_response_selection_mask = traj.response_selection_mask[:self.response_length]

                output = AgentLoopOutput(
                    prompt_ids=truncated_prompt_ids,
                    response_ids=truncated_response_ids,
                    response_mask=truncated_response_mask,
                    multi_modal_data=multi_modal_data_dict if i == len(agent_data.trajectory_outputs) - 1 else {},
                    response_logprobs=truncated_logprobs,
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    routed_experts=agent_data.routed_experts if i == len(agent_data.trajectory_outputs) - 1 else None,
                    extra_fields={
                        **agent_data.extra_fields,
                        "traj_idx": i,
                        "is_final": traj.is_final,
                        "overlong": agent_data.overlong,
                        "rollout_id": rollout_id,
                        "uid": uid,
                        "is_padding": is_padding,  # Pass through for SUPO unpad
                        "reward_model": reward_model,  # Pass through for reward calculation
                        "data_source": data_source,  # Pass through for metrics
                        "turn_scores": agent_data.turn_scores,
                        "tool_rewards": agent_data.tool_rewards,
                        "echo_selected_traj_indices": final_selected_traj_indices if traj.is_final else None,
                        "echo_selected_turn_ids": final_selected_turn_ids if traj.is_final else None,
                        "echo_memory_graph": memory_graph if traj.is_final else None,
                        "echo_response_turn_ids": truncated_response_turn_ids,
                        "echo_response_finding_turn_ids": truncated_response_finding_turn_ids,
                        "echo_response_selection_mask": truncated_response_selection_mask,
                        "echo_diagnostics": echo_diagnostics,
                    },
                )
                outputs.append(output)

            logger.debug(
                "Context-compressed rollout completed. "
                f"num_trajectories={len(outputs)}, overlong={agent_data.overlong}, "
                f"summary_count={agent_data.summary_count}"
            )


            return outputs

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    def _merge_generation_extra_fields(self, agent_data: AgentData, extra_fields: dict[str, Any]):
        """Propagate rollout-server metadata across multi-turn and compressed trajectories."""
        if not extra_fields:
            return

        for key, value in extra_fields.items():
            if key not in {"global_steps", "min_global_steps", "max_global_steps"}:
                agent_data.extra_fields[key] = value

        version_start = extra_fields.get("min_global_steps", extra_fields.get("global_steps"))
        version_end = extra_fields.get("max_global_steps", extra_fields.get("global_steps"))

        if version_start is not None:
            previous_start = agent_data.extra_fields.get("min_global_steps")
            agent_data.extra_fields["min_global_steps"] = (
                version_start if previous_start is None else min(previous_start, version_start)
            )

        if version_end is not None:
            agent_data.extra_fields["global_steps"] = version_end
            agent_data.extra_fields["max_global_steps"] = version_end

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        # Add the turn-summary requirement to the system prompt for turn-selection compression.
        if self.enable_summarization and self._uses_sum_last_turn_history():
            sum_instruction = "\n\n" + self.sum_last_turn_instruction.strip()
            for msg in agent_data.messages:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + sum_instruction
                    break
            else:
                agent_data.messages.insert(0, {"role": "system", "content": self.sum_last_turn_instruction.strip()})

        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.chat_template_tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids

        # Keep the original prompt so compressed contexts can be rebuilt from it.
        if self.enable_summarization:
            agent_data.original_prompt_ids = list(prompt_ids)
            agent_data.original_messages = list(agent_data.messages)
            # The current segment keeps history in the response side so generated tokens remain trainable.
            agent_data.current_traj_prompt_ids = list(prompt_ids)

        return AgentState.GENERATING


    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""

        with simple_timer("generate_sequences", agent_data.metrics):
            generate_prompt_ids = agent_data.prompt_ids
            generation_sampling_params = self._sampling_params_for_generation(sampling_params, agent_data)

            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=generate_prompt_ids,  # 使用拼接后的 prompt
                sampling_params=generation_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        self._merge_generation_extra_fields(agent_data, output.extra_fields)

        # Compression generations are auxiliary and do not count as assistant turns.
        if not (self.enable_summarization and agent_data.is_summarizing):
            agent_data.assistant_turns += 1

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids

        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Handle the auxiliary compression generation.
        if self.enable_summarization and agent_data.is_summarizing:
            self._append_summary_to_current_trajectory(agent_data, output.log_probs or [])

            full_output = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )

            compression_ok = True
            selected_indices: list = []
            model_selected_indices: list = []
            pending = None
            if self.context_compression_method == "echo_e2e":
                agent_data.echo_selection_response_lengths.append(len(agent_data.response_ids))
                agent_data.echo_selection_stop_reasons.append(str(output.stop_reason or ""))
                self._mark_last_trajectory_selection_tokens(agent_data, full_output)
                pending = getattr(agent_data, 'pending_turn', None)
                if pending:
                    finding = self._extract_turn_finding(full_output)
                    if finding:
                        pending["finding"] = finding
                    pending["text"] = self._build_turn_text(pending.get("action", ""), pending.get("finding", ""))
                    self._mark_last_trajectory_finding_tokens(agent_data, full_output, pending.get("turn_id"))

                if agent_data.turn_history:
                    parsed = self._parse_selection_indices(full_output, len(agent_data.turn_history))
                    if parsed is None:
                        agent_data.echo_selection_parse_failures += 1
                        output_snippet = re.sub(r"\s+", " ", full_output).strip()[:200]
                        logger.warning(
                            "ECHO selection parse failed; retaining all prior turns. "
                            f"num_turns={len(agent_data.turn_history)}, "
                            f"response_length={len(agent_data.response_ids)}, "
                            f"stop_reason={output.stop_reason!r}, output={output_snippet!r}"
                        )
                        agent_data.memory_graph_valid = False
                        context_selected = [
                            {"index": i, "score": 0.5, "source": "parse_fallback"}
                            for i in range(len(agent_data.turn_history))
                        ]
                    else:
                        model_selected_indices = parsed
                        context_selected = parsed
                    selected_indices = self._merge_selected_with_recent(
                        context_selected, len(agent_data.turn_history)
                    )
                    new_messages = self._build_echo_prompt(agent_data, selected_indices)
                else:
                    new_messages = self._build_echo_prompt(agent_data, [])
                    model_selected_indices = []
            elif self._is_truncate_method():
                logger.warning("Unexpected auxiliary generation for truncate context compression.")
                compression_ok = False
            else:
                summary_text = self._summary_exclude_think(full_output)
                if len(summary_text.strip()) < 10:
                    logger.warning(f"Context summary is too short; text={summary_text[:50]!r}")
                    compression_ok = False
                else:
                    new_messages = [
                        *agent_data.original_messages,
                        {"role": "assistant", "content": f"<summary>{summary_text}</summary>"},
                        {"role": "user", "content": "The above is a summary of your previous interaction history. Please continue solving the problem based on this context. You may call tools as needed."},
                    ]

            if not compression_ok:
                agent_data.is_summarizing = False
                agent_data.overlong = True
                logger.warning("Context compression output unusable. Terminating.")
                return AgentState.TERMINATED

            new_prompt_ids = await self.apply_chat_template(new_messages, tools=self.chat_template_tool_schemas)
            if self._is_turn_selection_method():
                agent_data.echo_reconstruction_prompt_lengths.append(len(new_prompt_ids))
                if len(new_prompt_ids) > self.prompt_length:
                    agent_data.echo_prompt_budget_overflow_count += 1
                agent_data.current_selected_traj_indices = sorted({
                    agent_data.turn_history[item["index"]].get("source_traj_idx")
                    for item in selected_indices
                    if item["index"] < len(agent_data.turn_history)
                    and agent_data.turn_history[item["index"]].get("source_traj_idx") is not None
                })
                agent_data.current_selected_turn_ids = sorted({
                    agent_data.turn_history[item["index"]].get("turn_id")
                    for item in selected_indices
                    if item["index"] < len(agent_data.turn_history)
                    and agent_data.turn_history[item["index"]].get("turn_id") is not None
                })
                agent_data.current_model_selected_turn_ids = sorted({
                    agent_data.turn_history[item["index"]].get("turn_id")
                    for item in model_selected_indices
                    if item["index"] < len(agent_data.turn_history)
                    and agent_data.turn_history[item["index"]].get("turn_id") is not None
                })
                if self.context_compression_method == "echo_e2e" and self.echo_graph_version == 2:
                    self._start_echo_selection_event(agent_data)
                # ``working_context_length`` is the boundary that triggers a
                # split, not a post-selection prompt cap. The reconstructed
                # prompt may legitimately be larger than 0.8 * L as long as
                # the rollout server's physical context can hold one token.
                prompt_hard_limit = self._generation_prompt_limit()
                if len(new_prompt_ids) >= prompt_hard_limit:
                    agent_data.is_summarizing = False
                    agent_data.overlong = True
                    logger.warning(
                        "Selection rebuilt prompt exceeds the physical context window. "
                        f"prompt_length={len(new_prompt_ids)}, max_model_len={prompt_hard_limit}"
                    )
                    return AgentState.TERMINATED

            agent_data.prompt_ids = list(new_prompt_ids)
            agent_data.messages = new_messages

            # Keep the complete reconstructed prompt for the next rollout
            # request, while making the serialized PPO prompt fit the fixed
            # prompt tensor budget. The suffix is carried as masked response
            # context so post-processing does not silently drop the task,
            # system instructions, or tool schema from the model input.
            agent_data.current_traj_prompt_ids = list(agent_data.prompt_ids)
            agent_data.response_ids = []
            agent_data.response_mask = []
            agent_data.response_logprobs = []
            agent_data.accumulated_response_ids = []
            agent_data.accumulated_response_mask = []
            agent_data.accumulated_logprobs = []
            agent_data.accumulated_response_turn_ids = []
            agent_data.accumulated_response_finding_turn_ids = []
            agent_data.accumulated_response_selection_mask = []

            # prompt_length is the tensor budget for the serialized prompt
            # field, not a limit on the context sent to the rollout server.
            if len(agent_data.current_traj_prompt_ids) > self.prompt_length:
                context_ids = agent_data.current_traj_prompt_ids[self.prompt_length:]
                agent_data.current_traj_prompt_ids = agent_data.current_traj_prompt_ids[:self.prompt_length]
                agent_data.accumulated_response_ids = list(context_ids)
                agent_data.accumulated_response_mask = [0] * len(context_ids)
                agent_data.accumulated_logprobs = [0.0] * len(context_ids)
                agent_data.accumulated_response_turn_ids = [-1] * len(context_ids)
                agent_data.accumulated_response_finding_turn_ids = [-1] * len(context_ids)
                agent_data.accumulated_response_selection_mask = [0] * len(context_ids)

            agent_data.is_summarizing = False
            agent_data.summary_count += 1
            # Keep the source-indexed archive persistent across reconstructions.
            # selected_indices controls only the active prompt view.
            if self.context_compression_method == "echo_e2e":
                if pending and pending.get("finding"):
                    pending["text"] = self._build_turn_text(pending.get("action", ""), pending.get("finding", ""))
                    pending.setdefault("embedding", None)
                    agent_data.turn_history.append(pending)
                # A tool response that crossed the boundary belongs to the
                # newly reconstructed segment's local context. It is finalized
                # into turn_history only after the next assistant response.
                agent_data.pending_turn = getattr(agent_data, "echo_pending_local_turn", None)
                agent_data.echo_pending_local_turn = None
                agent_data.echo_pending_local_messages = []
                # The previous segment remains in the archive, but its local
                # parent set is no longer active unless selection retained it.
                agent_data.current_segment_turn_ids = []
                if agent_data.pending_turn:
                    # This boundary-crossing turn is the first turn in the
                    # reconstructed segment. Replace its old local lineage with
                    # the explicit selector dependencies.
                    if self.echo_graph_version == 1:
                        agent_data.pending_turn.pop("parent_edges", None)
                        agent_data.pending_turn["parent_turn_ids"] = self._current_echo_parent_turn_ids(agent_data)
                    else:
                        agent_data.pending_turn["parent_edges"] = self._current_echo_parent_edges(agent_data)
                        self._set_current_selection_target(agent_data, agent_data.pending_turn.get("turn_id"))
                # The carried local turn is part of the prompt even though it
                # was not present in ``turn_history`` when selection ran. Add
                # it to the active-context snapshot so the next turn and the
                # terminal outcome can depend on it as well.
                if agent_data.pending_turn:
                    pending_id = agent_data.pending_turn.get("turn_id")
                    if pending_id is not None:
                        try:
                            pending_id = int(pending_id)
                        except (TypeError, ValueError):
                            pending_id = None
                    if pending_id is not None and pending_id not in agent_data.current_selected_turn_ids:
                        agent_data.current_selected_turn_ids.append(pending_id)
                        agent_data.current_selected_turn_ids.sort()

            return AgentState.GENERATING

        if self.enable_summarization:
            agent_data.accumulated_response_ids.extend(agent_data.response_ids)
            agent_data.accumulated_response_mask.extend([1] * len(agent_data.response_ids))
            agent_data.accumulated_logprobs.extend(output.log_probs or [0.0] * len(agent_data.response_ids))
            current_turn_id = agent_data.next_turn_id if self._is_turn_selection_method() else -1
            response_turn_ids = [current_turn_id] * len(agent_data.response_ids)
            response_finding_turn_ids = [-1] * len(agent_data.response_ids)
            if self._uses_sum_last_turn_history():
                full_text = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                )
                summary_span_mask = self._build_summary_span_mask(full_text, agent_data.response_ids)
                response_turn_ids = self._mask_summary_span_turn_ids(response_turn_ids, summary_span_mask)
                pending = getattr(agent_data, "pending_turn", None)
                if pending:
                    response_finding_turn_ids = self._build_finding_turn_ids(
                        full_text, agent_data.response_ids, pending.get("turn_id")
                    )
            agent_data.accumulated_response_turn_ids.extend(response_turn_ids)
            agent_data.accumulated_response_finding_turn_ids.extend(response_finding_turn_ids)
            agent_data.accumulated_response_selection_mask.extend([0] * len(agent_data.response_ids))

        # Extract tool calls. Keep a raw-text diagnostic for malformed or
        # truncated calls that the parser could not recover; this is distinct
        # from a normal assistant response with no tool action.
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        if getattr(self, "context_compression_method", None) == "echo_e2e" and not agent_data.tool_calls:
            def _decode_raw_response():
                try:
                    return self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
                except TypeError:
                    return self.tokenizer.decode(agent_data.response_ids)

            raw_response_text = await self.loop.run_in_executor(None, _decode_raw_response)
            if re.search(TOOL_CALL_MARKER_PATTERN, raw_response_text, flags=re.IGNORECASE | re.DOTALL):
                agent_data.echo_tool_parse_failures += 1
                logger.warning(
                    "ECHO response contained a tool-call marker but no parsed tool call; terminating this turn."
                )

        # Parse before enforcing turn limits.  The last allowed assistant turn
        # may contain the terminal tool call (or a final search/open_page call)
        # and must still be dispatched; the processing state terminates after
        # recording its observation.  The old ordering silently discarded that
        # action and made the ``is_last_turn`` branch below unreachable.
        turn_limit_reached = bool(
            (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        )
        if turn_limit_reached and not agent_data.tool_calls:
            if self.enable_summarization and self._uses_sum_last_turn_history() and agent_data.pending_turn:
                full_text = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                )
                self._finalize_pending_echo_turn(agent_data, full_text)
            return AgentState.TERMINATED

        # A response without another tool call can terminate immediately after
        # an ECHO reconstruction. Finalize the boundary turn before returning;
        # otherwise its summary tokens are tagged but the turn never enters the
        # memory graph used for historical credit assignment.
        if self.enable_summarization and self._uses_sum_last_turn_history() and not agent_data.tool_calls:
            full_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            self._finalize_pending_echo_turn(agent_data, full_text)

        # Compress when a generated thinking turn is truncated before a valid tool call.
        if output.stop_reason == "length" and not agent_data.tool_calls:
            if self.enable_summarization and not agent_data.is_summarizing \
                    and agent_data.summary_count < self.max_summary_rounds \
                    and len(agent_data.accumulated_response_ids) > 0:
                logger.warning(
                    "Generated turn was truncated before a tool call; rolling back and compressing. "
                    f"response_length={len(agent_data.response_ids)}"
                )
                if self.context_compression_method == "echo_e2e":
                    agent_data.echo_trigger_count += 1
                    agent_data.echo_trigger_reasons.append("generation_length")
                    agent_data.echo_trigger_lengths.append(
                        len(agent_data.current_traj_prompt_ids) + len(agent_data.accumulated_response_ids)
                    )
                    agent_data.echo_trigger_prompt_lengths.append(len(agent_data.current_traj_prompt_ids))
                    agent_data.echo_trigger_response_lengths.append(len(agent_data.accumulated_response_ids))
                    agent_data.echo_trigger_tool_lengths.append(0)
                    agent_data.echo_trigger_history_lengths.append(len(agent_data.turn_history))
                if self.context_compression_method == "echo_e2e":
                    self._rollback_current_echo_response(agent_data)
                else:
                    a_t_len = len(agent_data.response_ids)
                    if a_t_len > 0:
                        agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                        agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                        agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                        agent_data.accumulated_response_turn_ids = agent_data.accumulated_response_turn_ids[:-a_t_len]
                        agent_data.accumulated_response_finding_turn_ids = (
                            agent_data.accumulated_response_finding_turn_ids[:-a_t_len]
                        )
                        agent_data.accumulated_response_selection_mask = (
                            agent_data.accumulated_response_selection_mask[:-a_t_len]
                        )
                if self.context_compression_method == "echo_e2e":
                    self._save_current_trajectory_without_current_turn(agent_data)
                    if not self._has_echo_selection_history(agent_data):
                        agent_data.overlong = True
                        logger.warning(
                            "ECHO generation truncated before reusable turn history was available. Marking overlong."
                        )
                        return AgentState.TERMINATED
                    return await self._trigger_echo_selection(agent_data)
                elif self.context_compression_method == "semantic_selection":
                    agent_data.pending_turn = None
                    self._save_current_trajectory_without_current_turn(agent_data)
                    if not self._has_echo_selection_history(agent_data):
                        agent_data.overlong = True
                        logger.warning(
                            "Semantic selection generation truncated before reusable turn history was available. "
                            "Marking overlong."
                        )
                        return AgentState.TERMINATED
                    return await self._trigger_semantic_selection(agent_data)
                elif self._is_truncate_method():
                    self._save_current_trajectory_without_current_turn(agent_data)
                    rollback_prompt_ids = agent_data.current_traj_prompt_ids + agent_data.accumulated_response_ids
                    return self._trigger_truncation(agent_data, base_prompt_ids=rollback_prompt_ids)
                else:
                    self._save_current_trajectory_without_current_turn(agent_data)
                    return await self._trigger_summary(agent_data)
            else:
                agent_data.overlong = True
                return AgentState.TERMINATED

        response_budget_length = (
            len(agent_data.accumulated_response_ids)
            if self.enable_summarization
            else len(agent_data.response_mask)
        )
        # A response may exactly fill the PPO budget and still contain a
        # complete tool call.  Parse/execute that call first; the tool handler
        # will append any fitting observation and terminate when no budget
        # remains for another assistant turn.
        if not ignore_termination and response_budget_length >= self.response_length and not agent_data.tool_calls:
            return AgentState.TERMINATED

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED


    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        # The parser may return more calls than the configured execution cap.
        # Keep the metadata/action description aligned with the calls that are
        # actually sent to the environment.
        executed_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        tasks = []
        tool_call_names = []
        for tool_call in executed_tool_calls:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                 # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # Finalize the previous ECHO turn once the next assistant response is available.
        if self.enable_summarization and self._uses_sum_last_turn_history():
            full_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            self._finalize_pending_echo_turn(agent_data, full_text)

        # Add a separate ECHO hint message so chat-template boundaries remain intact.
        echo_inject_hint = (
            self.enable_summarization
            and self._uses_sum_last_turn_history()
        )
        if echo_inject_hint:
            agent_data.pending_turn = {
                "action": self._format_turn_action(executed_tool_calls),
                "finding": "",
                "observation": self._compact_observation_hint(add_messages),
                "text": "",
                "embedding": None,
                "source_traj_idx": len(agent_data.trajectory_outputs),
                "turn_id": agent_data.next_turn_id,
            }
            if self.echo_graph_version == 1:
                agent_data.pending_turn["parent_turn_ids"] = self._current_echo_parent_turn_ids(agent_data)
            else:
                agent_data.pending_turn["parent_edges"] = self._current_echo_parent_edges(agent_data)
            if not agent_data.current_segment_turn_ids:
                self._set_current_selection_target(agent_data, agent_data.next_turn_id)
            agent_data.next_turn_id += 1

            hint_msg = {
                "role": "user",
                "content": self.sum_last_turn_hint,
            }
            encode_messages = add_messages + [hint_msg]
        else:
            encode_messages = add_messages

        agent_data.messages.extend(encode_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            tool_response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
            if echo_inject_hint:
                hint_ids = await self.apply_chat_template([hint_msg], remove_system_prompt=True)
                tool_response_ids = tool_response_ids + hint_ids
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            tool_response_ids = await self.apply_chat_template(
                encode_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        is_finish_tool = any(name.lower() in ("finish", "stop", "submit") for name in tool_call_names)
        # ``user_turns`` is incremented after this tool response is encoded.
        # Use the projected value for compression decisions so an overflow on
        # the last allowed tool turn cannot enter another selector round.
        turn_limit_reached = bool(
            (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
            or (self.max_user_turns and agent_data.user_turns + 1 >= self.max_user_turns)
        )

        if self.enable_summarization:
            current_total_length = (
                len(agent_data.current_traj_prompt_ids) +
                len(agent_data.accumulated_response_ids) +
                len(tool_response_ids)
            )

            if self.context_compression_method == "echo_e2e":
                if current_total_length >= self.working_context_length:
                    agent_data.echo_trigger_count += 1
                    agent_data.echo_trigger_reasons.append("context_overflow")
                    agent_data.echo_trigger_lengths.append(current_total_length)
                    agent_data.echo_trigger_prompt_lengths.append(len(agent_data.current_traj_prompt_ids))
                    agent_data.echo_trigger_response_lengths.append(len(agent_data.accumulated_response_ids))
                    agent_data.echo_trigger_tool_lengths.append(len(tool_response_ids))
                    agent_data.echo_trigger_history_lengths.append(len(agent_data.turn_history))
                elif is_finish_tool:
                    agent_data.echo_finish_before_threshold_count += 1

            if current_total_length >= self.working_context_length and not is_finish_tool and not turn_limit_reached:
                if agent_data.summary_count < self.max_summary_rounds:
                    if self.context_compression_method == "echo_e2e":
                        # Preserve the complete action in the segment that caused
                        # the boundary.  The tool observation is carried as local
                        # context into the reconstructed segment; dropping the
                        # action here would make a later selection unable to route
                        # credit to the source turn's policy tokens.
                        agent_data.echo_pending_local_turn = deepcopy(agent_data.pending_turn)
                        agent_data.echo_pending_local_messages = deepcopy(add_messages)
                        agent_data.pending_turn = None
                        # The environment turn has happened; its observation
                        # is carried into the next local context while the
                        # generated action remains in this saved segment.
                        agent_data.user_turns += 1
                        if new_images_this_turn:
                            if agent_data.image_data is None:
                                agent_data.image_data = []
                            elif not isinstance(agent_data.image_data, list):
                                agent_data.image_data = [agent_data.image_data]
                            agent_data.image_data.extend(new_images_this_turn)
                        self._save_current_trajectory_without_current_turn(agent_data)
                        # An empty archive is still recoverable: the selector
                        # emits an empty block and the rebuilt prompt keeps the
                        # current action/observation as local context. Marking
                        # this state overlong would drop otherwise valid first
                        # turns whenever one large tool result crosses L.
                        return await self._trigger_echo_selection(agent_data)
                    elif self.context_compression_method == "semantic_selection":
                        n_hist = len(getattr(agent_data, 'turn_history', []))
                        if n_hist == 0:
                            agent_data.overlong = True
                            logger.warning(
                                "Semantic selection overflow occurred before reusable turn history was available. "
                                "Marking overlong."
                            )
                            self._save_current_trajectory_without_current_turn(agent_data)
                            return AgentState.TERMINATED

                        current_action_hint = self._format_turn_action(agent_data.tool_calls)
                        current_observation_hint = self._compact_observation_hint(add_messages)

                        a_t_len = len(agent_data.response_ids)
                        if a_t_len > 0:
                            agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                            agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                            agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                            agent_data.accumulated_response_turn_ids = agent_data.accumulated_response_turn_ids[:-a_t_len]
                            agent_data.accumulated_response_finding_turn_ids = (
                                agent_data.accumulated_response_finding_turn_ids[:-a_t_len]
                            )
                            agent_data.accumulated_response_selection_mask = (
                                agent_data.accumulated_response_selection_mask[:-a_t_len]
                            )
                        agent_data.pending_turn = None
                        self._save_current_trajectory_without_current_turn(agent_data)
                        return await self._trigger_semantic_selection(
                            agent_data,
                            current_action_hint=current_action_hint,
                            current_observation_hint=current_observation_hint,
                        )
                    elif self._is_truncate_method():
                        self._save_current_trajectory_with_extra_response(
                            agent_data,
                            extra_response_ids=tool_response_ids,
                        )
                        if new_images_this_turn:
                            if agent_data.image_data is None:
                                agent_data.image_data = []
                            elif not isinstance(agent_data.image_data, list):
                                agent_data.image_data = [agent_data.image_data]
                            for img in new_images_this_turn:
                                agent_data.image_data.append(img)
                        agent_data.user_turns += 1
                        return self._trigger_truncation(
                            agent_data,
                            extra_prompt_ids=tool_response_ids,
                        )
                    else:
                        self._save_current_trajectory_with_extra_response(
                            agent_data,
                            extra_response_ids=tool_response_ids,
                        )
                        a_t_len = len(agent_data.response_ids)
                        if a_t_len > 0:
                            agent_data.accumulated_response_ids = agent_data.accumulated_response_ids[:-a_t_len]
                            agent_data.accumulated_response_mask = agent_data.accumulated_response_mask[:-a_t_len]
                            agent_data.accumulated_logprobs = agent_data.accumulated_logprobs[:-a_t_len]
                            agent_data.accumulated_response_turn_ids = agent_data.accumulated_response_turn_ids[:-a_t_len]
                            agent_data.accumulated_response_finding_turn_ids = (
                                agent_data.accumulated_response_finding_turn_ids[:-a_t_len]
                            )
                            agent_data.accumulated_response_selection_mask = (
                                agent_data.accumulated_response_selection_mask[:-a_t_len]
                            )
                        return await self._trigger_summary(agent_data)
                else:
                    agent_data.overlong = True
                    logger.warning(
                        f"Reached max context compression rounds ({self.max_summary_rounds}). Marking overlong."
                    )
                    self._save_current_trajectory_with_extra_response(
                        agent_data,
                        extra_response_ids=tool_response_ids,
                    )
                    return AgentState.TERMINATED


        response_budget_length = (
            len(agent_data.accumulated_response_ids)
            if self.enable_summarization
            else len(agent_data.response_mask)
        )
        tool_response_ids, response_budget_exhausted = self._fit_tool_response_to_budget(
            tool_response_ids,
            response_budget_length=response_budget_length,
            response_length=self.response_length,
            is_finish_tool=is_finish_tool,
        )

        if self.enable_summarization and self._uses_full_observation_history():
            self._append_full_observation_turn(agent_data, executed_tool_calls, add_messages)

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        # Tool responses are part of the trajectory but do not receive policy gradients.
        if self.enable_summarization:
            agent_data.accumulated_response_ids.extend(tool_response_ids)
            agent_data.accumulated_response_mask.extend([0] * len(tool_response_ids))
            agent_data.accumulated_logprobs.extend([0.0] * len(tool_response_ids))
            tool_turn_id = (
                agent_data.pending_turn["turn_id"]
                if echo_inject_hint and getattr(agent_data, "pending_turn", None)
                else -1
            )
            agent_data.accumulated_response_turn_ids.extend([tool_turn_id] * len(tool_response_ids))
            agent_data.accumulated_response_finding_turn_ids.extend([-1] * len(tool_response_ids))
            agent_data.accumulated_response_selection_mask.extend([0] * len(tool_response_ids))

        agent_data.prompt_ids += tool_response_ids
        agent_data.response_mask += [0] * len(tool_response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(tool_response_ids)
        agent_data.user_turns += 1
        # Re-check limits after recording the environment turn.  In
        # particular, max_user_turns is incremented above, so the pre-tool
        # value of turn_limit_reached cannot be reused here.
        turn_limit_reached_after_tool = bool(
            (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        )
        if is_finish_tool or response_budget_exhausted or turn_limit_reached_after_tool:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    @staticmethod
    def _normalize_tool_arguments(tool_name: str, raw_args: Any) -> dict[str, Any]:
        """Normalize common malformed tool arguments into the dict shape tools expect."""
        if isinstance(raw_args, dict):
            return raw_args

        if raw_args is None:
            return {}

        # Some models emit a JSON object as a string in the arguments field.
        if isinstance(raw_args, str):
            text = raw_args.strip()
            if not text:
                return {}
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

            if tool_name == "search":
                return {"query": text}
            if tool_name == "open_page":
                return {"docid": text}
            if tool_name == "finish":
                return {"answer": text}

        # Accept the common single-call shape defensively.
        if isinstance(raw_args, list) and len(raw_args) == 1 and isinstance(raw_args[0], dict):
            return raw_args[0]

        return {}

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = str(tool_call.name).strip()
            if tool_name not in self.tools and tool_name.lower() in self.tools:
                tool_name = tool_name.lower()
            try:
                raw_tool_args = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                raw_tool_args = tool_call.arguments
            tool_args = self._normalize_tool_arguments(tool_name, raw_tool_args)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map

    def _infer_semantic_embed_url(self) -> str:
        """Reuse the BCP retrieval service for semantic turn-selection embeddings."""
        search_tool = self.tools.get("search") if hasattr(self, "tools") else None
        retrieval_url = str(getattr(search_tool, "retrieval_service_url", "") or "")
        if retrieval_url.endswith("/retrieve"):
            return retrieval_url[: -len("/retrieve")] + "/embed"
        if retrieval_url:
            return retrieval_url.rstrip("/") + "/embed"
        return "http://localhost:8000/embed"

    def _is_turn_selection_method(self) -> bool:
        """Whether this compression method maintains ECHO-style turn history."""
        return self.context_compression_method in ("echo_e2e", "semantic_selection")

    def _generation_prompt_limit(self) -> int:
        """Return the largest prompt that still leaves rollout generation space.

        SGLang's default ``max_new_tokens`` is bounded by
        ``prompt_length + response_length - len(prompt_ids)``.  A prompt can
        therefore fit ``max_model_len`` and still receive zero generated
        tokens when it exceeds the tensor budget.  Keep ECHO reconstruction
        and selector prompts under both limits, while tolerating lightweight
        test doubles that only define ``max_model_len``.
        """
        configured_limit = getattr(self, "max_model_len", None)
        configured_limit = int(configured_limit) if configured_limit is not None else None

        prompt_length = getattr(self, "prompt_length", None)
        response_length = getattr(self, "response_length", None)
        if prompt_length is not None and response_length is not None:
            sampling_budget = int(prompt_length) + int(response_length)
        else:
            sampling_budget = None

        if configured_limit is None:
            fallback = sampling_budget
            if fallback is None:
                fallback = getattr(self, "working_context_length", 1)
            return max(1, int(fallback))
        if sampling_budget is None:
            return max(1, configured_limit)
        return max(1, min(configured_limit, sampling_budget))

    def _selection_generation_budget(self) -> int:
        """Return the bounded decoder budget for one ECHO selection."""
        configured = max(1, int(getattr(self, "selection_max_new_tokens", 1024)))
        response_length = getattr(self, "response_length", None)
        if response_length is not None:
            configured = min(configured, max(1, int(response_length)))
        return configured

    def _selection_prompt_limit(self) -> int:
        """Largest selector prompt that still leaves its decoder budget available."""
        generation_budget = self._selection_generation_budget()
        limits = []

        max_model_len = getattr(self, "max_model_len", None)
        if max_model_len is not None:
            # SGLang reserves one additional position when computing the
            # maximum number of generated tokens.
            limits.append(int(max_model_len) - generation_budget - 1)

        prompt_length = getattr(self, "prompt_length", None)
        response_length = getattr(self, "response_length", None)
        if prompt_length is not None and response_length is not None:
            limits.append(int(prompt_length) + int(response_length) - generation_budget)

        if not limits:
            limits.append(self._generation_prompt_limit() - generation_budget)
        return max(0, min(limits))

    def _sampling_params_for_generation(
        self, sampling_params: dict[str, Any], agent_data: AgentData
    ) -> dict[str, Any]:
        """Use a finite, reserved decoder budget for auxiliary ECHO selection."""
        if not (
            self.enable_summarization
            and agent_data.is_summarizing
            and self.context_compression_method == "echo_e2e"
        ):
            return sampling_params

        selector_sampling_params = dict(sampling_params)
        selector_sampling_params["max_new_tokens"] = self._selection_generation_budget()
        return selector_sampling_params

    def _uses_full_observation_history(self) -> bool:
        return self.context_compression_method == "semantic_selection" and self.semantic_selection_full_observation

    def _uses_sum_last_turn_history(self) -> bool:
        return self._is_turn_selection_method() and not self._uses_full_observation_history()

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    def _get_original_user_query(self, agent_data: AgentData) -> str:
        """Return the original user task for semantic selection queries."""
        messages = agent_data.original_messages or agent_data.messages
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = self._message_content_to_text(msg.get("content")).strip()
                if text:
                    return text[:2000]
        return ""

    @staticmethod
    def _build_turn_text(action: str, finding: str) -> str:
        return f"Action: {str(action or '').strip()}\nFinding: {str(finding or '').strip()}".strip()

    def _compact_observation_hint(self, messages: list[dict[str, Any]]) -> str:
        """Extract a short hint from the current tool observation without committing it as history."""
        chunks = []
        for msg in messages or []:
            if msg.get("role") != "tool":
                continue
            text = self._message_content_to_text(msg.get("content")).strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    text = str(parsed.get("result") or parsed.get("error") or text)
            except Exception:
                pass
            chunks.append(re.sub(r"\s+", " ", text).strip())
        return "\n".join(chunks)[:SEMANTIC_SELECTION_HINT_CHARS]

    def _compact_full_observation(self, messages: list[dict[str, Any]]) -> str:
        """Extract raw tool observations for sel-full turn history."""
        chunks = []
        for msg in messages or []:
            if msg.get("role") != "tool":
                continue
            text = self._message_content_to_text(msg.get("content")).strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    text = str(parsed.get("result") or parsed.get("error") or text)
            except Exception:
                pass
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append(text)

        observation = "\n".join(chunks).strip()
        max_chars = max(0, int(self.semantic_selection_full_observation_max_chars))
        if max_chars > 0 and len(observation) > max_chars:
            observation = observation[:max_chars] + "...(truncated)"
        return observation

    def _append_full_observation_turn(
        self,
        agent_data: AgentData,
        tool_calls: list[FunctionCall],
        tool_messages: list[dict[str, Any]],
    ):
        """Record a semantic-selection turn directly from raw tool observations."""
        action = self._format_turn_action(tool_calls)
        observation = self._compact_full_observation(tool_messages)
        if not action and not observation:
            return

        turn_id = agent_data.next_turn_id
        turn = {
            "action": action,
            "finding": observation,
            "observation": observation,
            "text": self._build_turn_text(action, observation),
            "embedding": None,
            "source_traj_idx": len(agent_data.trajectory_outputs),
            "turn_id": turn_id,
            "parent_turn_ids": self._current_echo_parent_turn_ids(agent_data),
        }
        agent_data.turn_history.append(turn)
        agent_data.next_turn_id += 1

    def _build_semantic_selection_query(
        self,
        agent_data: AgentData,
        current_action_hint: str = "",
        current_observation_hint: str = "",
    ) -> str:
        """Build a short state query for matching against turn_history embeddings."""
        parts = []
        original_query = self._get_original_user_query(agent_data)
        if original_query:
            parts.append(f"Question: {original_query}")

        state_units = []
        if current_action_hint or current_observation_hint:
            state_units.append({
                "action": current_action_hint,
                "finding": current_observation_hint,
                "label": "Current",
            })

        remaining = max(0, SEMANTIC_SELECTION_QUERY_UNITS - len(state_units))
        if remaining:
            for turn in agent_data.turn_history[-remaining:]:
                state_units.append({
                    "action": turn.get("action", ""),
                    "finding": turn.get("finding", ""),
                    "label": "Recent",
                })

        for item in state_units:
            turn_text = self._build_turn_text(item.get("action", ""), item.get("finding", ""))
            if turn_text:
                parts.append(f"{item['label']}:\n{turn_text}")

        return "\n\n".join(parts).strip()

    async def _embed_semantic_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "texts": texts,
            "instruction": SEMANTIC_SELECTION_INSTRUCTION,
        }

        def _post_embed():
            resp = requests.post(
                self.semantic_embed_url,
                json=payload,
                timeout=SEMANTIC_SELECTION_EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                raise RuntimeError(data["error"])
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise RuntimeError(f"Invalid embedding response from {self.semantic_embed_url}")
            return embeddings

        return await self.loop.run_in_executor(None, _post_embed)

    @staticmethod
    def _embedding_dot(query_emb: list[float], turn_emb: list[float]) -> float:
        return float(sum(float(a) * float(b) for a, b in zip(query_emb, turn_emb)))

    async def _select_semantic_turns(
        self,
        agent_data: AgentData,
        current_action_hint: str = "",
        current_observation_hint: str = "",
    ) -> list[dict]:
        """Select turn_history entries by embedding similarity to the current state query."""
        query_text = self._build_semantic_selection_query(
            agent_data,
            current_action_hint=current_action_hint,
            current_observation_hint=current_observation_hint,
        )
        missing_indices = []
        missing_texts = []
        for idx, turn in enumerate(agent_data.turn_history):
            if not turn.get("text"):
                turn["text"] = self._build_turn_text(turn.get("action", ""), turn.get("finding", ""))
            if turn.get("embedding") is None:
                missing_indices.append(idx)
                missing_texts.append(turn["text"])

        embeddings = await self._embed_semantic_texts([query_text] + missing_texts)
        query_emb = embeddings[0]
        for idx, emb in zip(missing_indices, embeddings[1:]):
            agent_data.turn_history[idx]["embedding"] = emb

        scored = []
        for idx, turn in enumerate(agent_data.turn_history):
            emb = turn.get("embedding")
            if emb is None:
                continue
            scored.append({
                "index": idx,
                "score": self._embedding_dot(query_emb, emb),
                "source": "semantic",
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: min(SEMANTIC_SELECTION_TOPK, len(scored))]

    def _extract_turn_finding(self, full_text: str) -> str:
        """Extract the latest ECHO turn summary from generated text."""
        m = re.search(r'<sum_last_turn>(.*?)</sum_last_turn>', full_text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return self._sanitize_turn_finding(m.group(1))
        m = re.search(r'<sum_last_turn>(.*)', full_text, flags=re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            logger.warning("No closing </sum_last_turn> tag found in turn summary; using text after <sum_last_turn>.")
            return self._sanitize_turn_finding(m.group(1))
        return ""

    def _sanitize_turn_finding(self, content: str) -> str:
        """Keep only concise factual text from a generated ECHO turn summary."""
        content = re.sub(r'<think>.*?</think>', '', str(content or ''), flags=re.DOTALL).strip()
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL).strip()
        content = re.split(
            TOOL_CALL_MARKER_PATTERN,
            content,
            maxsplit=1,
            flags=re.IGNORECASE | re.DOTALL,
        )[0].strip()
        content = re.sub(r'</?sum_last_turn>', '', content, flags=re.IGNORECASE).strip()
        content = re.sub(r'\s+', ' ', content).strip()
        return content[:self.sum_last_turn_max_chars]

    @staticmethod
    def _current_echo_parent_turn_ids(agent_data: AgentData) -> list[int]:
        """Return the archived selection-level parent set for a new turn."""
        parent_ids: list[int] = []
        for source_ids in (
            getattr(agent_data, "current_selected_turn_ids", []) or [],
            getattr(agent_data, "current_segment_turn_ids", []) or [],
        ):
            for raw_id in source_ids:
                try:
                    turn_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if turn_id not in parent_ids:
                    parent_ids.append(turn_id)
        return parent_ids

    @staticmethod
    def _current_echo_parent_edges(agent_data: AgentData) -> list[dict[str, Any]]:
        """Return the minimal typed lineage for the next ECHO reasoning turn."""
        local_turn_ids = getattr(agent_data, "current_segment_turn_ids", []) or []
        if local_turn_ids:
            try:
                parent_id = int(local_turn_ids[-1])
            except (TypeError, ValueError):
                return []
            return [{"turn_id": parent_id, "type": "turn"}]

        parent_edges = []
        for raw_id in getattr(agent_data, "current_model_selected_turn_ids", []) or []:
            try:
                turn_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            edge = {"turn_id": turn_id, "type": "selection"}
            if edge not in parent_edges:
                parent_edges.append(edge)
        return parent_edges

    @staticmethod
    def _start_echo_selection_event(agent_data: AgentData):
        selection_id = int(agent_data.next_echo_selection_id)
        agent_data.next_echo_selection_id += 1
        agent_data.current_echo_selection_id = selection_id
        agent_data.echo_selection_events.append(
            {
                "selection_id": selection_id,
                "source_traj_idx": len(agent_data.trajectory_outputs) - 1,
                "downstream_turn_id": None,
            }
        )

    @staticmethod
    def _set_current_selection_target(agent_data: AgentData, turn_id: Any):
        selection_id = getattr(agent_data, "current_echo_selection_id", None)
        if selection_id is None:
            return
        try:
            turn_id = int(turn_id)
        except (TypeError, ValueError):
            return
        for event in reversed(getattr(agent_data, "echo_selection_events", [])):
            if event.get("selection_id") == selection_id:
                if event.get("downstream_turn_id") is None:
                    event["downstream_turn_id"] = turn_id
                return

    def _finalize_pending_echo_turn(self, agent_data: AgentData, full_text: str) -> bool:
        """Commit the latest tool turn once its following assistant turn ends.

        ECHO keeps the action/observation pair pending until the next assistant
        response supplies ``<sum_last_turn>``. A natural-language termination
        has no later tool-processing callback, so it must use this same commit
        path explicitly.
        """
        pending = getattr(agent_data, "pending_turn", None)
        if not pending:
            return False

        finding = str(pending.get("finding", "") or "").strip()
        if not finding:
            finding = self._extract_turn_finding(full_text)
            if not finding:
                agent_data.echo_missing_finding_count += 1
                finding = str(pending.get("observation", ""))[: self.sum_last_turn_max_chars]
        if finding:
            pending["finding"] = finding
        pending["text"] = self._build_turn_text(pending.get("action", ""), pending.get("finding", ""))
        pending.setdefault("embedding", None)
        self._mark_finding_turn_tokens(agent_data, full_text, pending.get("turn_id"))
        agent_data.turn_history.append(pending)
        try:
            turn_id = int(pending.get("turn_id"))
        except (TypeError, ValueError):
            turn_id = None
        if turn_id is not None and turn_id not in agent_data.current_segment_turn_ids:
            agent_data.current_segment_turn_ids.append(turn_id)
        agent_data.pending_turn = None
        return True

    def _has_echo_selection_history(self, agent_data: AgentData) -> bool:
        """Whether ECHO has at least one retained turn that can participate in selection."""
        return bool(agent_data.turn_history)

    def _parse_selection_indices(self, full_output: str, max_turns: int) -> list[dict] | None:
        """Parse selected ECHO turn indices from actor output."""
        cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
        selection_match = re.search(r'<selection>(.*?)</selection>', cleaned, flags=re.DOTALL | re.IGNORECASE)
        has_selection_tag = selection_match is not None
        if selection_match:
            content = selection_match.group(1).strip()
        else:
            open_match = re.search(r'<selection>(.*)', cleaned, flags=re.DOTALL | re.IGNORECASE)
            if open_match:
                has_selection_tag = True
                content = open_match.group(1).strip()
                logger.warning("No closing </selection> tag found in compression output; using text after <selection>.")
            else:
                content = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
                if content:
                    logger.warning("No <selection> tag found in compression output; using fallback text.")
                else:
                    logger.warning("Empty ECHO selection output without a <selection> tag.")
                    return None

        content = re.split(
            r'<\|FunctionCallBegin\||<tool_call>|<function_call>',
            content,
            maxsplit=1,
            flags=re.IGNORECASE | re.DOTALL,
        )[0].strip()
        results = []
        seen = set()
        out_of_range = []
        # Accept the documented one-turn-per-line format as well as compact
        # outputs such as ``turn_0, turn_2``.  The latter is common when the
        # selector emits a short empty/recent-turn decision in one line.
        turn_matches = re.finditer(r"\bturn_(\d+)\b", content, flags=re.IGNORECASE)
        for match in turn_matches:
            idx = int(match.group(1))
            if idx in seen:
                continue
            seen.add(idx)
            if idx < max_turns:
                results.append({"index": idx, "score": 0.5})
            else:
                out_of_range.append(idx)
        if self.selection_max_turns > 0 and len(results) > self.selection_max_turns:
            logger.warning(
                f"ECHO selection produced {len(results)} valid turns; "
                f"keeping first {self.selection_max_turns} model-selected turns before recent-turn retention."
            )
            results = results[:self.selection_max_turns]
        if out_of_range:
            logger.warning(
                f"ECHO selection contained out-of-range turns {out_of_range}; "
                f"max_turn_index={max_turns - 1}, valid_selections={[r['index'] for r in results]}"
            )
        if not has_selection_tag:
            logger.warning("ECHO selection output had no explicit <selection> block.")
            return None
        return sorted(results, key=lambda x: x["index"])

    @staticmethod
    def _get_turn_action(turn: dict) -> str:
        """Return the generic turn action, with compatibility for older query-only entries."""
        return str(turn.get("action") or turn.get("query") or "")

    @staticmethod
    def _format_single_tool_action(tool_call: FunctionCall) -> str:
        """Format a tool/function call into a compact task-agnostic action string."""
        name = str(tool_call.name)
        try:
            args = json.loads(tool_call.arguments)
        except Exception:
            return f"{name}({tool_call.arguments})"[:300]

        if not isinstance(args, dict):
            return f"{name}({args})"[:300]

        # BCP/search tasks benefit from showing the human-readable lookup target.
        if name == "search" and args.get("query"):
            return f"search(query={args.get('query')})"[:300]
        if name == "open_page":
            docid = args.get("docid") or args.get("id") or args.get("url")
            if docid:
                return f"open_page(docid={docid})"[:300]
        if name == "finish" and args.get("answer"):
            return f"finish(answer={args.get('answer')})"[:300]

        # CodeGym and other function-calling tasks should keep the actual function
        # name plus parameters instead of pretending the action is a search query.
        compact_args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        return f"{name}({compact_args})"[:300]

    def _format_turn_action(self, tool_calls: list[FunctionCall]) -> str:
        """Format one assistant turn's tool/function calls for ECHO turn history."""
        actions = [self._format_single_tool_action(tool_call) for tool_call in tool_calls]
        return " | ".join(action for action in actions if action)[:600]

    def _build_echo_local_messages(self, agent_data: AgentData) -> list[dict]:
        """Render the active turn retained across an ECHO reconstruction."""
        pending = getattr(agent_data, "echo_pending_local_turn", None)
        if not pending:
            return []

        messages = [
            {
                "role": "assistant",
                "content": f"[current turn Action] {self._get_turn_action(pending)}",
            },
            *deepcopy(getattr(agent_data, "echo_pending_local_messages", []) or []),
            {"role": "user", "content": self.sum_last_turn_hint},
        ]
        return messages

    def _format_echo_local_context(self, agent_data: AgentData) -> str:
        """Format the active action/observation for the selector instruction."""
        pending = getattr(agent_data, "echo_pending_local_turn", None)
        if not pending:
            return ""

        observation_parts = []
        for message in getattr(agent_data, "echo_pending_local_messages", []) or []:
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("text")
                )
            if content:
                observation_parts.append(str(content))

        action = self._get_turn_action(pending)
        observation = "\n".join(observation_parts)
        if not action and not observation:
            return ""
        return f"Action: {action}\nObservation:\n{observation}".strip()

    def _build_echo_prompt(self, agent_data: AgentData, selected_turns: list[dict]) -> list[dict]:
        """Build prompt messages from selected ECHO history and local context."""
        selected_messages = []
        for item in selected_turns:
            idx = item["index"]
            if idx < len(agent_data.turn_history):
                turn = agent_data.turn_history[idx]
                action = self._get_turn_action(turn)
                finding = turn.get("finding", "")
                selected_messages.append({"role": "assistant", "content": f"[turn_{idx} Action] {action}"})
                selected_messages.append({"role": "tool", "content": f"[turn_{idx} Key Finding] {finding}"})

        selected_messages.extend(self._build_echo_local_messages(agent_data))
        new_messages = [
            *agent_data.original_messages,
            *selected_messages,
            {"role": "user", "content": "The above are the most relevant turns from your previous interaction. Please continue solving the problem based on this context. You may call tools as needed."},
        ]
        selected_indices = [item["index"] for item in selected_turns]
        return new_messages

    @staticmethod
    def _build_echo_memory_graph(agent_data: AgentData) -> dict[str, Any] | None:
        """Serialize the sparse typed ECHO reasoning DAG for SUPO."""
        if not getattr(agent_data, "memory_graph_valid", True):
            return None

        nodes = []
        known_ids = set()
        # Read hand-built/old in-memory entries without changing the format
        # emitted by the current loop. Real ECHO turns always carry the
        # ``parent_edges`` key, so new rollouts are version 2.
        legacy_untyped_graph = getattr(agent_data, "echo_graph_version", 2) == 1 or (
            bool(getattr(agent_data, "turn_history", []))
            and not any("parent_edges" in turn for turn in getattr(agent_data, "turn_history", []))
            and not getattr(agent_data, "echo_selection_events", [])
        )
        for turn in getattr(agent_data, "turn_history", []):
            turn_id = turn.get("turn_id")
            if turn_id is None:
                continue
            try:
                turn_id = int(turn_id)
            except (TypeError, ValueError):
                continue
            if turn_id in known_ids:
                logger.warning("Ignoring duplicate ECHO Turn-Memory node turn_id=%s", turn_id)
                continue
            known_ids.add(turn_id)
            parent_edges = []
            seen_edges = set()
            raw_parent_edges = [] if legacy_untyped_graph else (turn.get("parent_edges", []) or [])
            if legacy_untyped_graph:
                raw_parent_edges = [
                    {"turn_id": raw_id, "type": "turn"}
                    for raw_id in turn.get("parent_turn_ids", []) or []
                ]
            for raw_edge in raw_parent_edges:
                if not isinstance(raw_edge, dict):
                    continue
                try:
                    parent_id = int(raw_edge.get("turn_id"))
                except (TypeError, ValueError):
                    continue
                edge_type = str(raw_edge.get("type", "") or "").lower()
                edge_key = (parent_id, edge_type)
                if parent_id == turn_id or edge_type not in {"turn", "selection"} or edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                parent_edges.append({"turn_id": parent_id, "type": edge_type})
            nodes.append(
                {
                    "turn_id": turn_id,
                    "source_traj_idx": turn.get("source_traj_idx"),
                    "parent_edges": parent_edges,
                }
            )

        if legacy_untyped_graph:
            outcome_parent_ids = []
            for raw_id in ToolAgentLoop._current_echo_parent_turn_ids(agent_data):
                try:
                    turn_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if turn_id in known_ids and turn_id not in outcome_parent_ids:
                    outcome_parent_ids.append(turn_id)
            return {
                "version": 1,
                "nodes": [
                    {
                        "turn_id": node["turn_id"],
                        "source_traj_idx": node["source_traj_idx"],
                        "parent_turn_ids": [edge["turn_id"] for edge in node["parent_edges"]],
                    }
                    for node in nodes
                ],
                "outcome_parent_turn_ids": sorted(outcome_parent_ids),
                "final_model_selected_turn_ids": sorted({
                    int(raw_id) for raw_id in getattr(agent_data, "current_model_selected_turn_ids", []) or []
                    if str(raw_id).lstrip("-").isdigit() and int(raw_id) in known_ids
                }),
            }

        outcome_parent_edges = []
        local_turn_ids = getattr(agent_data, "current_segment_turn_ids", []) or []
        if local_turn_ids:
            try:
                final_turn_id = int(local_turn_ids[-1])
            except (TypeError, ValueError):
                final_turn_id = None
            if final_turn_id in known_ids:
                outcome_parent_edges.append({"turn_id": final_turn_id, "type": "turn"})
        else:
            # A model may answer immediately after reconstruction. Then the
            # explicit selection is the only terminal provenance available.
            for raw_id in getattr(agent_data, "current_model_selected_turn_ids", []) or []:
                try:
                    turn_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if turn_id in known_ids:
                    outcome_parent_edges.append({"turn_id": turn_id, "type": "selection"})

        selection_events = []
        for raw_event in getattr(agent_data, "echo_selection_events", []) or []:
            try:
                selection_id = int(raw_event.get("selection_id"))
                source_traj_idx = int(raw_event.get("source_traj_idx"))
            except (TypeError, ValueError):
                continue
            downstream_turn_id = raw_event.get("downstream_turn_id")
            if downstream_turn_id is None and selection_id == getattr(agent_data, "current_echo_selection_id", None):
                downstream_turn_id = "outcome"
            elif downstream_turn_id is not None:
                try:
                    downstream_turn_id = int(downstream_turn_id)
                except (TypeError, ValueError):
                    continue
            selection_events.append(
                {
                    "selection_id": selection_id,
                    "source_traj_idx": source_traj_idx,
                    "downstream_turn_id": downstream_turn_id,
                }
            )

        nodes.sort(key=lambda node: node["turn_id"])
        return {
            "version": 2,
            "nodes": nodes,
            "outcome_parent_edges": outcome_parent_edges,
            "selection_events": selection_events,
        }

    def _merge_selected_with_recent(
        self,
        selected_turns: list[dict],
        max_turns: int,
        recent_k: int | None = None,
    ) -> list[dict]:
        """Retain model-selected turns plus the latest configured ECHO turns."""
        if max_turns <= 0:
            return []

        merged: dict[int, dict] = {}
        for item in selected_turns or []:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < max_turns:
                merged[idx] = item

        recent_k = self.echo_recent_turns if recent_k is None else recent_k
        recent_k = max(0, int(recent_k))
        if recent_k > 0:
            start = max(0, max_turns - recent_k)
            for idx in range(start, max_turns):
                merged.setdefault(idx, {"index": idx, "score": 0.5, "source": "recent"})

        return [merged[idx] for idx in sorted(merged)]

    def _summary_exclude_think(self, full_output):
        """Extract summary text while ignoring the generated thinking block."""
        without_closed_think = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()

        match = re.search(r'<summary>(.*?)</summary>', without_closed_think, flags=re.DOTALL)
        if match:
            summary_content = match.group(1).strip()
        else:
            open_match = re.search(r'<summary>(.*)', without_closed_think, flags=re.DOTALL)
            if open_match:
                summary_content = open_match.group(1).strip()
                logger.warning("No closing </summary> tag found in compression output; using text after <summary>.")
            else:
                # If the model forgot to close <think> but still started <summary>,
                # prefer the explicit summary span over dropping the whole output.
                open_match = re.search(r'<summary>(.*)', full_output, flags=re.DOTALL)
                if open_match:
                    summary_content = open_match.group(1).strip()
                    logger.warning("No closing </summary> tag found in compression output; using text after <summary>.")
                else:
                    cleaned = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
                    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
                    summary_content = cleaned
                    logger.warning("No <summary> tag found in compression output; using fallback text.")

        if len(summary_content) > self.summary_max_chars:
            summary_content = summary_content[:self.summary_max_chars] + "...(truncated)"
            logger.warning(f"Context summary truncated to {self.summary_max_chars} characters.")

        return summary_content

    def _is_truncate_method(self) -> bool:
        """Whether context compression should keep a left-truncated raw context."""
        return self.context_compression_method in ("truncate", "truncated")

    def _left_truncate_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        """Keep the most recent tokens under the configured truncate budget."""
        truncate_budget = max(1, int(self.working_context_length * self.truncate_keep_ratio))
        keep_len = min(len(prompt_ids), truncate_budget)
        if keep_len <= 0:
            return []
        return list(prompt_ids[-keep_len:])

    def _trigger_truncation(
        self,
        agent_data: AgentData,
        base_prompt_ids: list[int] | None = None,
        extra_prompt_ids: list[int] | None = None,
    ) -> AgentState:
        """Continue from the raw context suffix after saving the current trajectory segment."""
        full_prompt_ids = list(base_prompt_ids if base_prompt_ids is not None else agent_data.prompt_ids)
        if extra_prompt_ids:
            full_prompt_ids.extend(extra_prompt_ids)

        new_prompt_ids = self._left_truncate_prompt_ids(full_prompt_ids)
        agent_data.prompt_ids = new_prompt_ids
        agent_data.current_traj_prompt_ids = list(new_prompt_ids)
        agent_data.response_ids = []
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.accumulated_response_ids = []
        agent_data.accumulated_response_mask = []
        agent_data.accumulated_logprobs = []
        agent_data.accumulated_response_turn_ids = []
        agent_data.accumulated_response_finding_turn_ids = []
        agent_data.accumulated_response_selection_mask = []
        agent_data.is_summarizing = False
        agent_data.summary_count += 1
        logger.debug(
            "Triggering truncate context compression. "
            f"round={agent_data.summary_count}, prompt_length={len(new_prompt_ids)}, "
            f"truncate_keep_ratio={self.truncate_keep_ratio}"
        )
        return AgentState.GENERATING

    async def _trigger_semantic_selection(
        self,
        agent_data: AgentData,
        current_action_hint: str = "",
        current_observation_hint: str = "",
    ) -> AgentState:
        """Rebuild context with embedding-selected turn history."""
        try:
            selected = await self._select_semantic_turns(
                agent_data,
                current_action_hint=current_action_hint,
                current_observation_hint=current_observation_hint,
            )
        except Exception as exc:
            logger.warning(f"Semantic selection failed; retaining recent turns only. error={exc}")
            selected = []

        selected_indices = self._merge_selected_with_recent(
            selected,
            len(agent_data.turn_history),
            recent_k=SEMANTIC_SELECTION_RECENT_TURNS,
        )
        if not selected_indices:
            agent_data.overlong = True
            logger.warning("Semantic selection had no reusable turns. Marking overlong.")
            return AgentState.TERMINATED

        new_messages = self._build_echo_prompt(agent_data, selected_indices)
        new_prompt_ids = await self.apply_chat_template(new_messages, tools=self.chat_template_tool_schemas)
        prompt_hard_limit = self._generation_prompt_limit()
        if len(new_prompt_ids) >= prompt_hard_limit:
            agent_data.overlong = True
            logger.warning(
                "Semantic selection rebuilt prompt exceeds the physical context window. "
                f"prompt_length={len(new_prompt_ids)}, max_model_len={prompt_hard_limit}"
            )
            return AgentState.TERMINATED

        agent_data.current_selected_traj_indices = sorted({
            agent_data.turn_history[item["index"]].get("source_traj_idx")
            for item in selected_indices
            if item["index"] < len(agent_data.turn_history)
            and agent_data.turn_history[item["index"]].get("source_traj_idx") is not None
        })
        agent_data.current_selected_turn_ids = sorted({
            agent_data.turn_history[item["index"]].get("turn_id")
            for item in selected_indices
            if item["index"] < len(agent_data.turn_history)
            and agent_data.turn_history[item["index"]].get("turn_id") is not None
        })
        # Semantic selection has no model-generated <selection> block and is
        # therefore not used to create ECHO graph edges.
        agent_data.current_model_selected_turn_ids = []

        agent_data.prompt_ids = new_prompt_ids
        agent_data.messages = new_messages
        agent_data.current_traj_prompt_ids = list(agent_data.prompt_ids)
        agent_data.response_ids = []
        agent_data.response_logprobs = []
        agent_data.accumulated_response_ids = []
        agent_data.accumulated_response_mask = []
        agent_data.accumulated_logprobs = []
        agent_data.accumulated_response_turn_ids = []
        agent_data.accumulated_response_finding_turn_ids = []
        agent_data.accumulated_response_selection_mask = []
        agent_data.is_summarizing = False
        agent_data.summary_count += 1
        # selected_indices controls the active prompt view; the source-indexed
        # archive remains intact for later reconstruction boundaries.
        agent_data.pending_turn = None
        logger.debug(
            "Semantic selection compression complete. "
            f"selected_turns={[item['index'] for item in selected_indices]}, "
            f"prompt_length={len(new_prompt_ids)}"
        )
        return AgentState.GENERATING

    async def _trigger_summary(self, agent_data: AgentData) -> AgentState:
        """Trigger summary-based context compression."""
        summary_messages = [{"role": "user", "content": self.summary_instruction}]
        summary_instruction_ids = await self.apply_chat_template(
            summary_messages,
            remove_system_prompt=True,
            **self._compression_chat_template_overrides(),
        )
        agent_data.sum_instruction_ids = summary_instruction_ids
        agent_data.prompt_ids = (
            agent_data.current_traj_prompt_ids +
            agent_data.accumulated_response_ids +
            summary_instruction_ids
        )
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.is_summarizing = True
        logger.debug(f"Triggering summary compression. summary_count={agent_data.summary_count}")
        return AgentState.GENERATING

    async def _trigger_echo_selection(self, agent_data: AgentData) -> AgentState:
        """Trigger ECHO turn-selection context compression."""
        agent_data.echo_selection_count += 1
        turn_descriptions = []
        for i, turn in enumerate(agent_data.turn_history):
            action = self._get_turn_action(turn)
            finding = turn.get("finding", "")
            turn_descriptions.append(f"turn_{i}: action=\"{action}\" | finding=\"{finding}\"")
        turn_list_str = "\n".join(turn_descriptions) if turn_descriptions else "(none)"
        n_hist = len(agent_data.turn_history)

        instruction = self.selection_instruction.format(turn_list=turn_list_str)
        if n_hist > 0:
            instruction += f"\n\nValid selection indices: turn_0 ~ turn_{n_hist-1}."
            if self.selection_max_turns > 0:
                instruction += (
                    f"\nSelect at most {self.selection_max_turns} historical turns; "
                    "output an empty selection if the automatically retained recent turns are enough."
                )
            if self.echo_recent_turns > 0:
                recent_start = max(0, n_hist - self.echo_recent_turns)
                instruction += (
                    f"\nThe latest turns turn_{recent_start} ~ turn_{n_hist-1} "
                    "will be retained automatically; select any additional older turns that are still needed."
                )
        else:
            instruction += f"\n\nNo historical turns for selection, return empty inside <selection></selection>."

        local_context = self._format_echo_local_context(agent_data)
        if local_context:
            instruction += (
                "\n\nCurrent local context (the action and tool result that caused this boundary):\n"
                f"{local_context}\n"
                "Use this state when deciding which archived turns are still relevant."
            )

        # Keep the selection instruction separate from the task prompt.
        selection_messages = [{"role": "user", "content": instruction}]
        selection_instruction_ids = await self.apply_chat_template(
            selection_messages,
            remove_system_prompt=True,
            **self._compression_chat_template_overrides(),
        )

        current_prompt_ids = list(agent_data.current_traj_prompt_ids)
        accumulated_ids = list(agent_data.accumulated_response_ids)
        selection_prefix_ids = current_prompt_ids + accumulated_ids
        selector_prompt_limit = self._selection_prompt_limit()
        if selector_prompt_limit is not None:
            # Reserve the complete auxiliary decoder budget before retaining
            # rollout context. Without this reservation, a boundary near the
            # tensor limit leaves only a handful of tokens for <selection>.
            selector_prefix_budget = max(0, selector_prompt_limit - len(selection_instruction_ids))
            if len(selection_prefix_ids) > selector_prefix_budget:
                # Preserve the complete current segment prompt whenever it
                # fits. It contains the original task/system/tool schema;
                # discard only the oldest generated context first. A suffix
                # of the prompt is the last resort if the prompt itself is
                # larger than the physical selector budget.
                if len(current_prompt_ids) <= selector_prefix_budget:
                    response_budget = selector_prefix_budget - len(current_prompt_ids)
                    selection_prefix_ids = (
                        current_prompt_ids + accumulated_ids[-response_budget:]
                        if response_budget
                        else current_prompt_ids
                    )
                else:
                    selection_prefix_ids = current_prompt_ids[-selector_prefix_budget:] if selector_prefix_budget else []
                agent_data.echo_prompt_budget_overflow_count += 1

        agent_data.echo_selection_prefix_ids = list(selection_prefix_ids)
        agent_data.prompt_ids = selection_prefix_ids + selection_instruction_ids
        agent_data.sum_instruction_ids = selection_instruction_ids
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.is_summarizing = True

        return AgentState.GENERATING

    @staticmethod
    def _echo_diagnostics(agent_data: AgentData) -> dict[str, Any]:
        """Return JSON-safe rollout diagnostics for offline split analysis."""
        return {
            "trigger_count": int(agent_data.echo_trigger_count),
            "trigger_reasons": list(agent_data.echo_trigger_reasons),
            "trigger_lengths": list(agent_data.echo_trigger_lengths),
            "trigger_prompt_lengths": list(agent_data.echo_trigger_prompt_lengths),
            "trigger_response_lengths": list(agent_data.echo_trigger_response_lengths),
            "trigger_tool_lengths": list(agent_data.echo_trigger_tool_lengths),
            "trigger_history_lengths": list(agent_data.echo_trigger_history_lengths),
            "selection_count": int(agent_data.echo_selection_count),
            "selection_parse_failures": int(agent_data.echo_selection_parse_failures),
            "selection_response_lengths": list(agent_data.echo_selection_response_lengths),
            "selection_stop_reasons": list(agent_data.echo_selection_stop_reasons),
            "tool_parse_failures": int(agent_data.echo_tool_parse_failures),
            "missing_finding_count": int(agent_data.echo_missing_finding_count),
            "finish_before_threshold_count": int(agent_data.echo_finish_before_threshold_count),
            "reconstruction_prompt_lengths": list(agent_data.echo_reconstruction_prompt_lengths),
            "prompt_budget_overflow_count": int(agent_data.echo_prompt_budget_overflow_count),
            "summary_count": int(agent_data.summary_count),
            "trajectory_count": int(len(agent_data.trajectory_outputs)),
            "turn_history_count": int(len(agent_data.turn_history)),
            "overlong": bool(agent_data.overlong),
        }

    def _compression_chat_template_overrides(self) -> dict[str, bool]:
        """Disable thinking only for auxiliary compression prompts when supported."""
        if "enable_thinking" in self.apply_chat_template_kwargs:
            return {"enable_thinking": False}
        return {}

    # SUPO/ECHO support
    def _build_text_span_mask(
        self,
        text: str,
        response_ids: list[int],
        span_start: int,
        span_end: int,
    ) -> list[bool]:
        """Map a character span back to generated tokens without BPE drift."""
        mask = [False] * len(response_ids)
        if span_end <= span_start or not response_ids:
            return mask

        # Fast tokenizers expose offsets on the full string. Tokenizing the
        # prefix and span independently can change boundary merges (for
        # example, a trailing space before ``<selection>``), shifting masks by
        # one token. Use offsets only when the decoded text round-trips to the
        # same number of generated ids; special-token outputs use the fallback.
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            full_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
            offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
            if full_ids and offsets and len(full_ids) == len(response_ids):
                for idx, (token_start, token_end) in enumerate(offsets):
                    if token_end > span_start and token_start < span_end:
                        mask[idx] = True
                return mask
        except (TypeError, ValueError, AttributeError, KeyError, NotImplementedError, RuntimeError):
            pass

        prefix_ids = self.tokenizer.encode(text[:span_start], add_special_tokens=False)
        span_ids = self.tokenizer.encode(text[span_start:span_end], add_special_tokens=False)
        start = min(len(prefix_ids), len(response_ids))
        end = min(start + len(span_ids), len(response_ids))
        for idx in range(start, end):
            mask[idx] = True
        return mask

    def _build_summary_span_mask(self, text: str, response_ids: list[int]) -> list[bool]:
        """Identify generated <sum_last_turn> tokens in the current response."""
        if not response_ids:
            return []
        match = re.search(r"<sum_last_turn>.*?</sum_last_turn>", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            span_start = match.start()
            span_end = match.end()
        else:
            open_match = re.search(r"<sum_last_turn>(.*)", text, flags=re.DOTALL | re.IGNORECASE)
            if not open_match:
                return [False] * len(response_ids)
            span_start = open_match.start()
            marker_match = re.search(TOOL_CALL_MARKER_PATTERN, open_match.group(1), flags=re.IGNORECASE | re.DOTALL)
            if marker_match:
                span_end = open_match.start(1) + marker_match.start()
            else:
                span_end = len(text)
        if span_end <= span_start:
            return [False] * len(response_ids)
        return self._build_text_span_mask(text, response_ids, span_start, span_end)

    def _build_selection_span_mask(self, text: str, response_ids: list[int]) -> list[bool]:
        """Identify generated <selection> tokens in an ECHO compression response."""
        if not response_ids:
            return []

        match = re.search(r"<selection>.*?</selection>", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            span_start = match.start()
            span_end = match.end()
        else:
            open_match = re.search(r"<selection>(.*)", text, flags=re.DOTALL | re.IGNORECASE)
            if not open_match:
                return [False] * len(response_ids)
            span_start = open_match.start()
            marker_match = re.search(TOOL_CALL_MARKER_PATTERN, open_match.group(1), flags=re.IGNORECASE | re.DOTALL)
            if marker_match:
                span_end = open_match.start(1) + marker_match.start()
            else:
                span_end = len(text)
        if span_end <= span_start:
            return [False] * len(response_ids)
        return self._build_text_span_mask(text, response_ids, span_start, span_end)

    def _build_finding_turn_ids(self, text: str, response_ids: list[int], turn_id: int | None) -> list[int]:
        """Map the generated <sum_last_turn> span to the contributing ECHO turn."""
        finding_turn_ids = [-1] * len(response_ids)
        if turn_id is None:
            return finding_turn_ids

        summary_span_mask = self._build_summary_span_mask(text, response_ids)
        for idx, is_summary_token in enumerate(summary_span_mask):
            if is_summary_token:
                finding_turn_ids[idx] = int(turn_id)
        return finding_turn_ids

    @staticmethod
    def _mask_summary_span_turn_ids(response_turn_ids: list[int], summary_span_mask: list[bool]) -> list[int]:
        """Remove current-turn credit from summary tokens that belong to a previous turn."""
        return [
            -1 if idx < len(summary_span_mask) and summary_span_mask[idx] else turn_id
            for idx, turn_id in enumerate(response_turn_ids)
        ]

    def _mark_finding_turn_tokens(self, agent_data: AgentData, text: str, turn_id: int | None):
        """Mark the current assistant response's summary tokens for ECHO credit assignment."""
        finding_turn_ids = self._build_finding_turn_ids(text, agent_data.response_ids, turn_id)
        if not any(turn_idx >= 0 for turn_idx in finding_turn_ids):
            return

        start = len(agent_data.accumulated_response_finding_turn_ids) - len(agent_data.response_ids)
        if start < 0:
            return

        for offset, turn_idx in enumerate(finding_turn_ids):
            target_idx = start + offset
            if 0 <= target_idx < len(agent_data.accumulated_response_finding_turn_ids):
                agent_data.accumulated_response_finding_turn_ids[target_idx] = turn_idx
                if target_idx < len(agent_data.accumulated_response_turn_ids):
                    agent_data.accumulated_response_turn_ids[target_idx] = -1

    def _mark_last_trajectory_finding_tokens(self, agent_data: AgentData, text: str, turn_id: int | None):
        """Mark summary tokens appended by a compression generation."""
        if not agent_data.trajectory_outputs:
            return

        finding_turn_ids = self._build_finding_turn_ids(text, agent_data.response_ids, turn_id)
        if not any(turn_idx >= 0 for turn_idx in finding_turn_ids):
            return

        last_traj = agent_data.trajectory_outputs[-1]
        start = len(last_traj.response_finding_turn_ids) - len(agent_data.response_ids)
        if start < 0:
            return

        for offset, turn_idx in enumerate(finding_turn_ids):
            target_idx = start + offset
            if 0 <= target_idx < len(last_traj.response_finding_turn_ids):
                last_traj.response_finding_turn_ids[target_idx] = turn_idx

    def _mark_last_trajectory_selection_tokens(self, agent_data: AgentData, text: str):
        """Mark selection tokens appended by an ECHO compression generation."""
        if not agent_data.trajectory_outputs:
            return

        selection_span_mask = self._build_selection_span_mask(text, agent_data.response_ids)
        if not any(selection_span_mask):
            return

        last_traj = agent_data.trajectory_outputs[-1]
        start = len(last_traj.response_selection_mask) - len(agent_data.response_ids)
        if start < 0:
            return

        for offset, is_selection_token in enumerate(selection_span_mask):
            if not is_selection_token:
                continue
            target_idx = start + offset
            if 0 <= target_idx < len(last_traj.response_selection_mask):
                last_traj.response_selection_mask[target_idx] = 1

    @staticmethod
    def _rollback_current_echo_response(agent_data: AgentData):
        """Rollback an incomplete generation while retaining its finding.

        This is used when generation stops at the token limit before a valid
        tool call exists.  The assistant response can still contain the
        previous turn's ``<sum_last_turn>`` finding, which is reusable memory;
        arbitrary incomplete reasoning/action text is discarded.
        """
        response_len = len(agent_data.response_ids)
        if response_len <= 0:
            return

        fields = (
            "accumulated_response_ids",
            "accumulated_response_mask",
            "accumulated_logprobs",
            "accumulated_response_turn_ids",
            "accumulated_response_finding_turn_ids",
            "accumulated_response_selection_mask",
        )
        finding_ids = getattr(agent_data, "accumulated_response_finding_turn_ids", [])
        start = len(finding_ids) - response_len
        if start < 0:
            # The metadata should be aligned, but fall back to the historical
            # behavior instead of retaining an incorrectly aligned suffix.
            for field_name in fields:
                values = getattr(agent_data, field_name)
                setattr(agent_data, field_name, values[:-response_len])
            return

        finding_offsets = [
            idx for idx in range(start, len(finding_ids))
            if finding_ids[idx] is not None and int(finding_ids[idx]) >= 0
        ]
        if not finding_offsets:
            for field_name in fields:
                values = getattr(agent_data, field_name)
                setattr(agent_data, field_name, values[:-response_len])
            return

        for field_name in fields:
            values = getattr(agent_data, field_name)
            prefix = values[:start]
            finding_suffix = [values[idx] for idx in finding_offsets if idx < len(values)]
            setattr(agent_data, field_name, prefix + finding_suffix)

    @staticmethod
    def _fit_tool_response_to_budget(
        tool_response_ids: list[int],
        *,
        response_budget_length: int,
        response_length: int,
        is_finish_tool: bool,
    ) -> tuple[list[int], bool]:
        """Fit an observation into the serialized PPO response budget.

        A complete tool call must be processed even when the assistant has
        already consumed the response budget.  In that case a finish result is
        unnecessary, while a regular observation is kept up to the remaining
        capacity before the loop terminates.
        """
        if response_budget_length + len(tool_response_ids) < response_length:
            return tool_response_ids, False

        remaining_budget = max(0, response_length - response_budget_length)
        if is_finish_tool:
            return [], True
        return tool_response_ids[:remaining_budget], True

    def _save_current_trajectory_without_current_turn(self, agent_data: AgentData):
        """Save the accumulated segment after the caller has chosen its boundary."""
        resp = agent_data.accumulated_response_ids
        mask = agent_data.accumulated_response_mask
        logp = agent_data.accumulated_logprobs
        turn_ids = agent_data.accumulated_response_turn_ids
        finding_turn_ids = agent_data.accumulated_response_finding_turn_ids
        selection_mask = agent_data.accumulated_response_selection_mask

        min_len = min(len(resp), len(mask), len(logp), len(turn_ids), len(finding_turn_ids), len(selection_mask))
        if min_len == 0:
            return

        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=list(agent_data.current_traj_prompt_ids),
            response_ids=list(resp[:min_len]),
            response_mask=list(mask[:min_len]),
            response_logprobs=list(logp[:min_len]),
            response_turn_ids=list(turn_ids[:min_len]),
            response_finding_turn_ids=list(finding_turn_ids[:min_len]),
            response_selection_mask=list(selection_mask[:min_len]),
            is_final=False,
        ))


    def _save_current_trajectory_with_extra_response(
        self,
        agent_data: AgentData,
        extra_response_ids: list[int] | None = None,
        extra_response_turn_id: int = -1,
    ):
        """Save the current segment, optionally appending a just-produced tool response.

        When a tool response makes the segment cross the working-context limit,
        the next compressed state discards that overflow turn. The policy tokens
        from the turn should still remain trainable, so we persist them here
        before rolling back accumulated state for summary generation.
        """
        resp = list(agent_data.accumulated_response_ids)
        mask = list(agent_data.accumulated_response_mask)
        logp = list(agent_data.accumulated_logprobs)
        turn_ids = list(agent_data.accumulated_response_turn_ids)
        finding_turn_ids = list(agent_data.accumulated_response_finding_turn_ids)
        selection_mask = list(agent_data.accumulated_response_selection_mask)

        if extra_response_ids:
            extra_len = len(extra_response_ids)
            resp.extend(extra_response_ids)
            mask.extend([0] * extra_len)
            logp.extend([0.0] * extra_len)
            turn_ids.extend([extra_response_turn_id] * extra_len)
            finding_turn_ids.extend([-1] * extra_len)
            selection_mask.extend([0] * extra_len)

        min_len = min(len(resp), len(mask), len(logp), len(turn_ids), len(finding_turn_ids), len(selection_mask))
        if min_len == 0:
            return

        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=list(agent_data.current_traj_prompt_ids),
            response_ids=list(resp[:min_len]),
            response_mask=list(mask[:min_len]),
            response_logprobs=list(logp[:min_len]),
            response_turn_ids=list(turn_ids[:min_len]),
            response_finding_turn_ids=list(finding_turn_ids[:min_len]),
            response_selection_mask=list(selection_mask[:min_len]),
            is_final=False,
        ))


    def _append_summary_to_current_trajectory(self, agent_data: AgentData, summary_logprobs: list[float]):
        """Store trainable compression output with its exact sampling prefix."""
        # A standard SUPO summary is generated after the overflow assistant turn
        # has been rolled back from accumulated_response_ids. Keep the overflow
        # segment trainable, but store the summary in a separate segment whose
        # token sequence exactly reconstructs the prompt used for generation.
        if not self._is_turn_selection_method():
            prefix_ids = list(agent_data.accumulated_response_ids)
            instruction_ids = list(getattr(agent_data, "sum_instruction_ids", []) or [])
            full_summary_prompt = list(agent_data.current_traj_prompt_ids) + prefix_ids + instruction_ids
            summary_prompt_ids = full_summary_prompt[:self.prompt_length]
            summary_context_ids = full_summary_prompt[self.prompt_length:]
            summary_ids = list(agent_data.response_ids)
            summary_len = len(summary_ids)
            summary_logprobs = list(summary_logprobs[:summary_len])
            if len(summary_logprobs) < summary_len:
                summary_logprobs.extend([0.0] * (summary_len - len(summary_logprobs)))

            context_len = len(summary_context_ids)
            response_ids = summary_context_ids + summary_ids
            summary_loss_mask = int(getattr(self, "train_summary_tokens", True))
            agent_data.trajectory_outputs.append(TrajectoryOutput(
                prompt_ids=summary_prompt_ids,
                response_ids=response_ids,
                response_mask=[0] * context_len + [summary_loss_mask] * summary_len,
                response_logprobs=[0.0] * context_len + summary_logprobs,
                response_turn_ids=[-1] * len(response_ids),
                response_finding_turn_ids=[-1] * len(response_ids),
                response_selection_mask=[0] * len(response_ids),
                is_final=False,
            ))
            return

        # Selection is a policy-generated action in its own right.  Store it
        # as a separate segment with the selector prompt as masked context;
        # appending it to the previous response can right-truncate the
        # selection when that segment is already near the PPO response limit.
        selector_prefix_ids = getattr(agent_data, "echo_selection_prefix_ids", None)
        if selector_prefix_ids is None:
            selector_prefix_ids = list(agent_data.current_traj_prompt_ids) + list(agent_data.accumulated_response_ids)
        prefix_ids = list(selector_prefix_ids) + list(getattr(agent_data, "sum_instruction_ids", []) or [])
        prefix_prompt_ids = prefix_ids[: self.prompt_length]
        prefix_context_ids = prefix_ids[self.prompt_length :]
        selection_ids = list(agent_data.response_ids)
        selection_logprobs = list(summary_logprobs[: len(selection_ids)])
        if len(selection_logprobs) < len(selection_ids):
            selection_logprobs.extend([0.0] * (len(selection_ids) - len(selection_logprobs)))
        max_context_len = max(0, self.response_length - len(selection_ids))
        if len(prefix_context_ids) > max_context_len:
            # The selector prompt can be longer than the fixed PPO response
            # tensor because it contains the pre-boundary rollout plus the
            # selection instruction.  Keep a suffix of that prompt so the
            # policy-generated selection itself is retained and trainable.
            serialized_prefix = prefix_ids[-(self.prompt_length + max_context_len) :]
            prefix_prompt_ids = serialized_prefix[: self.prompt_length]
            prefix_context_ids = serialized_prefix[self.prompt_length :]
            agent_data.echo_prompt_budget_overflow_count += 1
        response_ids = prefix_context_ids + selection_ids
        context_len = len(prefix_context_ids)
        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=prefix_prompt_ids,
            response_ids=response_ids,
            response_mask=[0] * context_len + [1] * len(selection_ids),
            response_logprobs=[0.0] * context_len + selection_logprobs,
            response_turn_ids=[-1] * len(response_ids),
            response_finding_turn_ids=[-1] * len(response_ids),
            response_selection_mask=[0] * len(response_ids),
            is_final=False,
        ))

    def _save_final_trajectory(self, agent_data: AgentData):
        """Save the final trajectory segment."""
        min_len = min(
            len(agent_data.accumulated_response_ids),
            len(agent_data.accumulated_response_mask),
            len(agent_data.accumulated_logprobs),
            len(agent_data.accumulated_response_turn_ids),
            len(agent_data.accumulated_response_finding_turn_ids),
            len(agent_data.accumulated_response_selection_mask),
        )
        agent_data.trajectory_outputs.append(TrajectoryOutput(
            prompt_ids=list(agent_data.current_traj_prompt_ids),
            response_ids=list(agent_data.accumulated_response_ids[:min_len]),
            response_mask=list(agent_data.accumulated_response_mask[:min_len]),
            response_logprobs=list(agent_data.accumulated_logprobs[:min_len]),
            response_turn_ids=list(agent_data.accumulated_response_turn_ids[:min_len]),
            response_finding_turn_ids=list(agent_data.accumulated_response_finding_turn_ids[:min_len]),
            response_selection_mask=list(agent_data.accumulated_response_selection_mask[:min_len]),
            is_final=True,
        ))

        logger.debug(
            f"Saved final trajectory segment {len(agent_data.trajectory_outputs)-1}. "
            f"prompt_len={len(agent_data.current_traj_prompt_ids)}, "
            f"response_len={len(agent_data.accumulated_response_ids)}"
        )
