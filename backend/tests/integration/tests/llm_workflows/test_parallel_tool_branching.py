import json
from typing import Any
from uuid import uuid4

from onyx.configs.constants import DocumentSource
from onyx.server.query_and_chat.streaming_models import StreamingType
from onyx.tools.constants import SEARCH_TOOL_ID
from tests.integration.common_utils.managers.cc_pair import CCPairManager
from tests.integration.common_utils.managers.chat import ChatSessionManager
from tests.integration.common_utils.managers.mock_llm import MockLLMScript
from tests.integration.common_utils.managers.persona import PersonaManager
from tests.integration.common_utils.managers.tool import ToolManager
from tests.integration.common_utils.test_models import DATestUser
from tests.integration.mock_services.mock_llm_server.models import (
    Reply,
    RequestConditions,
    ToolCall,
)

_BRANCHING = StreamingType.TOP_LEVEL_BRANCHING.value
_SEARCH_TOOL_NAME = "internal_search"
_UNKNOWN_TOOL_NAME = "tool_that_does_not_exist"


def _create_connector(admin_user: DATestUser) -> None:
    # internal_search is only offered when a non-default connector exists.
    CCPairManager.create_from_scratch(
        source=DocumentSource.INGESTION_API,
        user_performing_action=admin_user,
    )


def _search_calls() -> list[ToolCall]:
    return [
        ToolCall(
            id="call_search_alpha",
            name=_SEARCH_TOOL_NAME,
            arguments={"queries": ["alpha"]},
        ),
        ToolCall(
            id="call_search_beta",
            name=_SEARCH_TOOL_NAME,
            arguments={"queries": ["beta"]},
        ),
    ]


def _packet_indices(packets: list[dict[str, Any]], packet_type: str) -> list[int]:
    return [
        i for i, packet in enumerate(packets) if packet["obj"]["type"] == packet_type
    ]


def test_merged_searches_and_unknown_tool_do_not_branch(
    admin_user: DATestUser, mock_llm: MockLLMScript
) -> None:
    _create_connector(admin_user)
    mock_llm.conversation(
        "chat",
        Reply(
            tool_calls=[
                *_search_calls(),
                ToolCall(id="call_unknown", name=_UNKNOWN_TOOL_NAME, arguments={}),
            ],
            conditions=RequestConditions(offers=[_SEARCH_TOOL_NAME]),
        ),
        Reply(
            text="Merged answer.",
            conditions=RequestConditions(has_results_for=["call_search_alpha"]),
        ),
    )
    chat_session = ChatSessionManager.create(user_performing_action=admin_user)

    response = ChatSessionManager.send_message(
        chat_session_id=chat_session.id,
        message="what is the answer?",
        user_performing_action=admin_user,
    )

    assert response.error is None, f"Unexpected stream error: {response.error}"
    assert [entry.tool_call_id for entry in response.tool_call_debug] == [
        "call_search_alpha",
        "call_search_beta",
        "call_unknown",
    ]

    packets = response.packets
    assert _packet_indices(packets, _BRANCHING) == []

    search_starts = _packet_indices(packets, StreamingType.SEARCH_TOOL_START.value)
    assert len(search_starts) == 1
    queries = [
        query
        for i in _packet_indices(packets, StreamingType.SEARCH_TOOL_QUERIES_DELTA.value)
        for query in packets[i]["obj"]["queries"]
    ]
    assert "alpha" in queries
    assert "beta" in queries

    assert response.full_message == "Merged answer."

    tool_request, answer_request = mock_llm.requests_in("chat")
    assert _UNKNOWN_TOOL_NAME not in tool_request.tools
    assert answer_request.tool_result_ids() == ["call_search_alpha"]


def test_distinct_executed_tools_branch_before_tool_starts(
    admin_user: DATestUser, mock_llm: MockLLMScript, mock_llm_server: str
) -> None:
    _create_connector(admin_user)

    custom_tool_name = f"branching_ping_{uuid4().hex[:8]}"
    # The mock LLM server is a real HTTP server in the test process.
    custom_tool_id = ToolManager.create_custom(
        name=custom_tool_name,
        definition={
            "openapi": "3.0.0",
            "info": {
                "title": "Branching ping",
                "description": "Health check",
                "version": "1.0.0",
            },
            "servers": [{"url": mock_llm_server}],
            "paths": {
                "/health": {
                    "get": {
                        "summary": "Check server health",
                        "operationId": custom_tool_name,
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        },
        user_performing_action=admin_user,
    )
    search_tool = ToolManager.get_by_in_code_id(
        SEARCH_TOOL_ID, user_performing_action=admin_user
    )
    assert search_tool is not None, "SearchTool must exist for this test"
    persona = PersonaManager.create(
        tool_ids=[search_tool.id, custom_tool_id],
        user_performing_action=admin_user,
    )
    chat_session = ChatSessionManager.create(
        persona_id=persona.id, user_performing_action=admin_user
    )
    mock_llm.conversation(
        "chat",
        Reply(
            tool_calls=[
                *_search_calls(),
                ToolCall(id="call_ping", name=custom_tool_name, arguments={}),
            ],
            conditions=RequestConditions(offers=[_SEARCH_TOOL_NAME, custom_tool_name]),
        ),
        Reply(
            text="Both tools ran.",
            conditions=RequestConditions(
                has_results_for=["call_search_alpha", "call_ping"]
            ),
        ),
    )

    response = ChatSessionManager.send_message(
        chat_session_id=chat_session.id,
        message="search and ping",
        user_performing_action=admin_user,
    )

    assert response.error is None, f"Unexpected stream error: {response.error}"

    packets = response.packets
    branching = _packet_indices(packets, _BRANCHING)
    assert len(branching) == 1
    branching_packet = packets[branching[0]]
    assert branching_packet["obj"]["num_parallel_branches"] == 2

    search_starts = _packet_indices(packets, StreamingType.SEARCH_TOOL_START.value)
    custom_starts = _packet_indices(packets, StreamingType.CUSTOM_TOOL_START.value)
    assert len(search_starts) == 1
    assert len(custom_starts) == 1
    custom_deltas = _packet_indices(packets, StreamingType.CUSTOM_TOOL_DELTA.value)
    assert len(custom_deltas) == 1
    custom_delta = packets[custom_deltas[0]]["obj"]
    assert custom_delta["error"] is None
    assert custom_delta["data"] == {"status": "ok"}
    for start in search_starts + custom_starts:
        assert branching[0] < start
        assert (
            packets[start]["placement"]["turn_index"]
            == branching_packet["placement"]["turn_index"]
        )

    assert response.full_message == "Both tools ran."

    _, answer_request = mock_llm.requests_in("chat")
    assert sorted(answer_request.tool_result_ids()) == [
        "call_ping",
        "call_search_alpha",
    ]
    ping_result = answer_request.tool_result("call_ping")
    assert ping_result is not None
    assert json.loads(ping_result) == {"status": "ok"}
