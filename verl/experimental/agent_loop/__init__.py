# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import sys

from .agent_loop import AgentLoopBase, AgentLoopManager, AgentLoopWorker, AsyncLLMServerManager

# Importing agent.tool_agent_loop starts by importing this package's base module.
# Avoid importing it again while that module is still being initialized.
_AGENT_MODULE_LOADING = "agent.tool_agent_loop" in sys.modules
if not _AGENT_MODULE_LOADING:
    from agent.tool_agent_loop import ToolAgentLoop
    from .codegym_agent_loop import CodeGymAgentLoop
    from .single_turn_agent_loop import SingleTurnAgentLoop

    _ = [SingleTurnAgentLoop, ToolAgentLoop, CodeGymAgentLoop]


__all__ = ["AgentLoopBase", "AgentLoopManager", "AsyncLLMServerManager", "AgentLoopWorker", "ToolAgentLoop"]


def __getattr__(name: str):
    if name == "ToolAgentLoop":
        from agent.tool_agent_loop import ToolAgentLoop

        return ToolAgentLoop
    if name == "SingleTurnAgentLoop":
        from .single_turn_agent_loop import SingleTurnAgentLoop

        return SingleTurnAgentLoop
    if name == "CodeGymAgentLoop":
        from .codegym_agent_loop import CodeGymAgentLoop

        return CodeGymAgentLoop
    raise AttributeError(name)
