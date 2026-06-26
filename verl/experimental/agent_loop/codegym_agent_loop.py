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
import time
from typing import Any, Optional
from urllib import error, parse, request
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CodeGymClient:
    """Small synchronous client for the CodeGym online_server HTTP API."""

    def __init__(self, manager_host: str, env_str: str, session_id: str, timeout: float = 30.0):
        self.manager_host = manager_host.rstrip("/")
        self.env_str = env_str
        self.session_id = session_id
        self.timeout = timeout
        self.uid: Optional[str] = None
        self.env_host: Optional[str] = None

    def start(self) -> str:
        instance = self._acquire_instance()
        self.uid = str(instance["uid"])
        port = str(instance["port"])
        self.env_host = self._host_with_port(self.manager_host, port)
        try:
            data = self._request_json(
                "POST",
                f"{self.env_host}/start",
                {"session_id": self.session_id, "env_str": self._server_env_str(), "env_name": "codegym_v1"},
                timeout=120.0,
            )
        except Exception:
            self.release()
            raise
        return str(data.get("observation", "env start successfully"))

    def _acquire_instance(self, timeout: float = 300.0, interval: float = 1.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                return self._request_json("GET", f"{self.manager_host}/get_instance")
            except error.HTTPError as e:
                last_error = e
                if e.code != 500:
                    raise
            time.sleep(interval)
        raise TimeoutError(f"Timed out acquiring CodeGym instance from {self.manager_host}: {last_error}")

    def step(self, action: str, timeout: float = 10.0) -> tuple[bool, str]:
        self._require_env_host()
        data = self._request_json(
            "POST",
            f"{self.env_host}/step",
            {"session_id": self.session_id, "action": action},
            timeout=timeout,
        )
        return bool(data.get("status", False)), str(data.get("observation", ""))

    def is_done(self) -> bool:
        self._require_env_host()
        data = self._request_json("GET", f"{self.env_host}/is_done?session_id={parse.quote(self.session_id)}")
        return bool(data.get("done", True))

    def reward(self) -> float:
        self._require_env_host()
        data = self._request_json("GET", f"{self.env_host}/reward?session_id={parse.quote(self.session_id)}")
        return float(data.get("reward", 0.0))

    def release(self) -> None:
        if self.env_host:
            try:
                self._request_json("POST", f"{self.env_host}/stop", {"session_id": self.session_id}, timeout=10.0)
            except Exception as e:
                logger.warning(f"Failed to stop CodeGym session: {e}")
        if self.uid:
            try:
                url = f"{self.manager_host}/release_instance?uid={parse.quote(self.uid)}"
                self._request_json("GET", url, timeout=10.0)
            except Exception as e:
                logger.warning(f"Failed to release CodeGym instance: {e}")

    def _server_env_str(self) -> str:
        if self.env_str.startswith("codegym_v1@"):
            return self.env_str[len("codegym_v1@") :]
        prefix, sep, payload = self.env_str.partition("@")
        if sep and "__" not in prefix:
            return f"codegym_v1__{prefix}@{payload}"
        return self.env_str

    @staticmethod
    def _host_with_port(host: str, port: str) -> str:
        parsed = parse.urlsplit(host)
        hostname = parsed.hostname or host
        scheme = parsed.scheme or "http"
        if ":" in hostname and not hostname.startswith("["):
            netloc = f"[{hostname}]:{port}"
        else:
            netloc = f"{hostname}:{port}"
        return parse.urlunsplit((scheme, netloc, "", "", ""))

    def _require_env_host(self) -> None:
        if not self.env_host:
            raise RuntimeError("CodeGym environment is not started")

    def _request_json(
        self,
        method: str,
        url: str,
        data: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        body = None
        headers = {}
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=body, headers=headers, method=method)
        with request.urlopen(req, timeout=timeout or self.timeout) as resp:
            payload = resp.read().decode("utf-8")
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}


@register("codegym_agent")
class CodeGymAgentLoop(ToolAgentLoop):
    """Tool-agent loop that routes dynamic CodeGym function calls to a remote environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codegym_client: Optional[CodeGymClient] = None
        self.codegym_done = False
        self.codegym_reward = 0.0

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        extra_info = kwargs.get("extra_info", {}) or {}
        env_str = extra_info.get("env_str") or kwargs.get("ability")
        if not env_str:
            raise ValueError("CodeGymAgentLoop requires extra_info.env_str or ability")

        server_host = (
            extra_info.get("codegym_server_host")
            or getattr(self.rollout_config.multi_turn, "codegym_server_host", None)
            or os.environ.get("CODEGYM_SERVER_HOST")
            or os.environ.get("CODEGYM_MANAGER_HOST")
        )
        if not server_host:
            raise ValueError("CodeGym server host is required via extra_info.codegym_server_host, config, or env")

        request_id = f"{kwargs.get('uid') or kwargs.get('index') or os.getpid()}-{uuid4().hex}"
        self.codegym_client = CodeGymClient(server_host, env_str, session_id=request_id)
        self.codegym_done = False
        self.codegym_reward = 0.0

        try:
            await self.loop.run_in_executor(None, self.codegym_client.start)
            output = await super().run(sampling_params, **kwargs)
            reward = self.codegym_reward
            if self.codegym_client is not None:
                try:
                    reward = await self.loop.run_in_executor(None, self.codegym_client.reward)
                except Exception as e:
                    logger.warning(f"Failed to fetch CodeGym reward: {e}")
            self._attach_final_reward(output, reward)
            return output
        finally:
            if self.codegym_client is not None:
                await self.loop.run_in_executor(None, self.codegym_client.release)
                self.codegym_client = None

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        if self.codegym_client is None:
            return ToolResponse(text="CodeGym environment is not initialized."), 0.0, {}

        try:
            parameters = json.loads(tool_call.arguments)
            action = json.dumps({"name": tool_call.name, "parameters": parameters}, ensure_ascii=False)
        except Exception as e:
            return ToolResponse(text=f"The action cannot be parsed in json format: {e}"), 0.0, {}

        try:
            status, observation = await self.loop.run_in_executor(None, self.codegym_client.step, action)
            done = await self.loop.run_in_executor(None, self.codegym_client.is_done)
            # CodeGym env.step returns a success flag for executing the action, not episode termination.
            # The actual terminal signal is exposed by /is_done.
            self.codegym_done = bool(done)
            if self.codegym_done:
                self.codegym_reward = await self.loop.run_in_executor(None, self.codegym_client.reward)
            return ToolResponse(text=observation), 0.0, {"codegym_done": self.codegym_done}
        except Exception as e:
            logger.warning(f"Error when executing CodeGym action: {e}")
            self.codegym_done = True
            return ToolResponse(text=f"Error when executing CodeGym action: {e}"), 0.0, {"codegym_error": str(e)}

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        next_state = await super()._handle_processing_tools_state(agent_data)
        if self.codegym_done:
            return AgentState.TERMINATED
        return next_state

    @staticmethod
    def _attach_final_reward(output: AgentLoopOutput | list[AgentLoopOutput], reward: float) -> None:
        if isinstance(output, list):
            if not output:
                return
            for segment in output:
                segment.reward_score = 0.0
            output[-1].reward_score = float(reward)
            output[-1].extra_fields.setdefault("reward_extra_info", {})["acc"] = float(reward > 0)
            output[-1].extra_fields["reward_extra_info"]["score"] = float(reward)
        else:
            output.reward_score = float(reward)
            output.extra_fields.setdefault("reward_extra_info", {})["acc"] = float(reward > 0)
            output.extra_fields["reward_extra_info"]["score"] = float(reward)
