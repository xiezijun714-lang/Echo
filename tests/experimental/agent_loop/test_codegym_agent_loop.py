import asyncio
import json

import pytest

from verl.experimental.agent_loop.codegym_agent_loop import CodeGymAgentLoop, CodeGymClient
from verl.experimental.agent_loop.tool_parser import ToolParser


class DummyTokenizer:
    def decode(self, ids, *args, **kwargs):
        return ids


def test_codegym_fc_parser_extracts_single_call():
    parser = ToolParser.get_tool_parser("codegym_fc", DummyTokenizer())
    text = '<|FunctionCallBegin|>[{"name":"Observe","parameters":{}}]<|FunctionCallEnd|>'

    content, calls = asyncio.run(parser.extract_tool_calls(text))

    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "Observe"
    assert json.loads(calls[0].arguments) == {}


def test_codegym_fc_parser_ignores_malformed_call():
    parser = ToolParser.get_tool_parser("codegym_fc", DummyTokenizer())
    text = "<|FunctionCallBegin|>[bad json]<|FunctionCallEnd|>"

    content, calls = asyncio.run(parser.extract_tool_calls(text))

    assert content == ""
    assert calls == []


def test_codegym_client_strips_dataset_namespace_prefix():
    client = CodeGymClient(
        manager_host="http://127.0.0.1:8000",
        env_str='codegym_v1@Leetcode_1_I__TwoSumEnv@{"nums":[1,2],"target":3}',
        session_id="sid",
    )

    assert client._server_env_str() == 'Leetcode_1_I__TwoSumEnv@{"nums":[1,2],"target":3}'


def test_codegym_client_strips_source_prefix_without_namespace():
    client = CodeGymClient(
        manager_host="http://127.0.0.1:8000",
        env_str='Codeforces_25150_I__AnagramGroupingEnv@{"strings":["a"]}',
        session_id="sid",
    )

    assert client._server_env_str() == 'Codeforces_25150_I__AnagramGroupingEnv@{"strings":["a"]}'


def test_codegym_client_adds_server_prefix_for_bare_env_name():
    client = CodeGymClient(
        manager_host="http://127.0.0.1:8000",
        env_str='RemoveDuplicatesEnv@{"input_string":"aa"}',
        session_id="sid",
    )

    assert client._server_env_str() == 'codegym_v1__RemoveDuplicatesEnv@{"input_string":"aa"}'


def test_attach_final_reward_sets_only_last_segment():
    agent_loop = pytest.importorskip("verl.experimental.agent_loop.agent_loop")
    first = agent_loop.AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        reward_score=None,
        metrics=agent_loop.AgentLoopMetrics(),
        extra_fields={},
    )
    last = first.model_copy(deep=True)

    CodeGymAgentLoop._attach_final_reward([first, last], 1.0)

    assert first.reward_score == 0.0
    assert last.reward_score == 1.0
    assert last.extra_fields["reward_extra_info"]["acc"] == 1.0
