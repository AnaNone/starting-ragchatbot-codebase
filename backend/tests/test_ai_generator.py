import pytest
import httpx
import anthropic
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator


def _text_response(text: str) -> MagicMock:
    """Build a mock Anthropic response with stop_reason=end_turn."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def _tool_use_response(name: str, tool_id: str, tool_input: dict) -> MagicMock:
    """Build a mock Anthropic response with stop_reason=tool_use."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = tool_input
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


@pytest.fixture
def mock_client():
    with patch("ai_generator.anthropic.Anthropic") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def gen(mock_client):
    return AIGenerator(api_key="test-key", model="claude-test-model")


def test_direct_text_no_tool(gen, mock_client):
    """When stop_reason=end_turn, returns content[0].text with exactly one API call."""
    mock_client.messages.create.return_value = _text_response(
        "The learning rate controls gradient descent step size."
    )
    result = gen.generate_response(query="What is learning rate?")
    assert result == "The learning rate controls gradient descent step size."
    mock_client.messages.create.assert_called_once()


def test_tool_use_calls_tool_manager(gen, mock_client):
    """When stop_reason=tool_use, tool_manager.execute_tool() is called with correct name and input."""
    mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "tu_001", {"query": "neural networks"}),
        _text_response("Neural networks are inspired by the brain."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "[ML Intro - Lesson 1]\nNeural networks content..."

    result = gen.generate_response(
        query="Tell me about neural networks",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="neural networks"
    )
    assert result == "Neural networks are inspired by the brain."


def test_tool_result_in_second_api_call(gen, mock_client):
    """The tool result string is included in the second API call as a tool_result message."""
    mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "tu_xyz789", {"query": "transformers"}),
        _text_response("Transformers use self-attention mechanisms."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "Transformers content from vector store"

    gen.generate_response(
        query="What are transformers?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    last_message = messages[-1]
    assert last_message["role"] == "user"
    content = last_message["content"]
    assert len(content) == 1
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "tu_xyz789"
    assert content[0]["content"] == "Transformers content from vector store"


def test_tools_passed_in_first_call(gen, mock_client):
    """tools= list and tool_choice=auto are included in the first API call."""
    tool_def = {
        "name": "search_course_content",
        "description": "Search course materials",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }
    mock_client.messages.create.return_value = _text_response("Direct answer.")

    gen.generate_response(query="What is ML?", tools=[tool_def])

    first_call_kwargs = mock_client.messages.create.call_args_list[0][1]
    assert "tools" in first_call_kwargs
    assert first_call_kwargs["tools"] == [tool_def]
    assert first_call_kwargs.get("tool_choice") == {"type": "auto"}


def test_sequential_two_round_tool_use(gen, mock_client):
    """Claude can call a tool, see the results, then call a second tool before answering."""
    mock_client.messages.create.side_effect = [
        _tool_use_response("get_course_outline", "tu_001", {"course_name": "ML Intro"}),
        _tool_use_response("search_course_content", "tu_002", {"query": "backpropagation", "lesson_number": 3}),
        _text_response("Lesson 3 covers backpropagation in detail."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = [
        "Course: ML Intro\nLesson 3: Backpropagation",
        "[ML Intro - Lesson 3]\nBackpropagation explanation...",
    ]

    result = gen.generate_response(
        query="Tell me about backpropagation in the ML Intro course",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
    tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="ML Intro")
    tool_manager.execute_tool.assert_any_call("search_course_content", query="backpropagation", lesson_number=3)
    assert result == "Lesson 3 covers backpropagation in detail."


def test_messages_accumulate_across_rounds(gen, mock_client):
    """Every round's tool_use/tool_result messages are threaded into later API calls."""
    mock_client.messages.create.side_effect = [
        _tool_use_response("get_course_outline", "tu_a", {"course_name": "ML Intro"}),
        _tool_use_response("search_course_content", "tu_b", {"query": "gradient descent"}),
        _text_response("Final synthesized answer."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["outline result", "search result"]

    gen.generate_response(
        query="Explain gradient descent in ML Intro",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    final_call_kwargs = mock_client.messages.create.call_args_list[2][1]
    messages = final_call_kwargs["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"][0]["tool_use_id"] == "tu_a"
    assert messages[2]["content"][0]["content"] == "outline result"
    assert messages[4]["content"][0]["tool_use_id"] == "tu_b"
    assert messages[4]["content"][0]["content"] == "search result"


def test_stops_as_soon_as_claude_is_done(gen, mock_client):
    """The loop exits the moment stop_reason != tool_use, without spending the full round budget."""
    mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "tu_001", {"query": "transformers"}),
        _text_response("Done after a single round."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search result"

    result = gen.generate_response(
        query="What are transformers?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_client.messages.create.call_count == 2
    assert result == "Done after a single round."


def test_max_rounds_forces_final_call_without_tools(gen, mock_client):
    """Once max_tool_rounds is reached, the final API call omits tools, forcing a text answer."""
    mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "tu_001", {"query": "round 1"}),
        _tool_use_response("search_course_content", "tu_002", {"query": "round 2"}),
        _text_response("Best-effort answer after exhausting tool rounds."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search result"

    result = gen.generate_response(
        query="A query that needs many lookups",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_client.messages.create.call_count == gen.max_tool_rounds + 1
    last_call_kwargs = mock_client.messages.create.call_args_list[-1][1]
    assert "tools" not in last_call_kwargs
    assert result == "Best-effort answer after exhausting tool rounds."


def test_hard_tool_error_wraps_up_with_is_error(gen, mock_client):
    """An exception from execute_tool is surfaced as an is_error tool_result and the loop stops immediately."""
    mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", "tu_001", {"query": "transformers"}),
        _text_response("I couldn't search right now, but here's what I know..."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("vector store unavailable")

    result = gen.generate_response(
        query="What are transformers?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert mock_client.messages.create.call_count == 2
    second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
    tool_result = second_call_kwargs["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is True
    assert "vector store unavailable" in tool_result["content"]
    assert result == "I couldn't search right now, but here's what I know..."


def test_empty_tool_results_returns_fallback(gen, mock_client):
    """stop_reason=tool_use but no tool_use blocks in content returns a fallback string, no AttributeError."""
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = []  # no tool_use blocks — _execute_tool_round yields empty tool_results
    mock_client.messages.create.return_value = resp
    tool_manager = MagicMock()

    result = gen.generate_response(
        query="What is ML?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "I was unable to process your request."
    mock_client.messages.create.assert_called_once()
    tool_manager.execute_tool.assert_not_called()


def test_anthropic_exception_propagates(gen, mock_client):
    """
    When client.messages.create() raises an Anthropic exception, it propagates
    uncaught out of generate_response(). This confirms the secondary bug:
    the exception travels through rag_system.query() → app.py → HTTP 500
    → frontend shows 'Query failed'.
    Fix 3 adds try/except in RAGSystem.query() to catch this.
    """
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(401, request=req, text="Unauthorized")
    mock_client.messages.create.side_effect = anthropic.AuthenticationError(
        message="Invalid API key",
        response=resp,
        body={"error": {"message": "Invalid API key"}},
    )
    with pytest.raises(anthropic.AuthenticationError):
        gen.generate_response(query="What is ML?")
