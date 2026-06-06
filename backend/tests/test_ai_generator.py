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
