import pytest
import httpx
import anthropic
import chromadb
from unittest.mock import MagicMock, patch


@pytest.fixture
def rag(tmp_path):
    """
    RAGSystem with in-memory ChromaDB and mocked Anthropic client.
    Yields (system, mock_generate) where mock_generate controls AI responses.
    """
    with (
        patch("vector_store.chromadb.PersistentClient") as mock_pc,
        patch("ai_generator.anthropic.Anthropic"),
    ):
        mock_pc.return_value = chromadb.EphemeralClient()

        from rag_system import RAGSystem

        cfg = MagicMock()
        cfg.CHROMA_PATH = str(tmp_path)
        cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        cfg.MAX_RESULTS = 5
        cfg.ANTHROPIC_API_KEY = "test-key"
        cfg.ANTHROPIC_MODEL = "claude-test-model"
        cfg.MAX_HISTORY = 2
        cfg.CHUNK_SIZE = 800
        cfg.CHUNK_OVERLAP = 100

        system = RAGSystem(cfg)
        mock_generate = MagicMock(return_value="Mocked AI answer")
        system.ai_generator.generate_response = mock_generate
        yield system, mock_generate


def test_query_returns_tuple_str_list(rag):
    """query() returns (str, list) — the contract expected by app.py."""
    system, _ = rag
    result = system.query("What is deep learning?")
    assert isinstance(result, tuple) and len(result) == 2
    answer, sources = result
    assert isinstance(answer, str)
    assert isinstance(sources, list)


def test_query_no_raise_when_tool_error(rag):
    """
    When the AI returns an error-narrating string (MAX_RESULTS=0 scenario),
    query() completes without raising — no HTTP 500.
    """
    system, mock_generate = rag
    mock_generate.return_value = (
        "I encountered a search error and cannot answer your question right now."
    )
    answer, sources = system.query("What is supervised learning?")
    assert isinstance(answer, str)
    assert isinstance(sources, list)


def test_query_returns_graceful_message_on_anthropic_exception(rag):
    """
    When AIGenerator raises an Anthropic exception (e.g. invalid API key),
    RAGSystem.query() now catches it and returns a graceful error message
    instead of propagating — eliminating the HTTP 500 / 'Query failed' path.
    """
    system, mock_generate = rag
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(401, request=req, text="Unauthorized")
    mock_generate.side_effect = anthropic.AuthenticationError(
        message="Invalid API key",
        response=resp,
        body={"error": {"message": "Invalid API key"}},
    )
    answer, sources = system.query("What is ML?")
    assert isinstance(answer, str)
    assert "Sorry" in answer or "couldn't" in answer
    assert sources == []


def test_session_history_updated_after_query(rag):
    """After query(session_id=sid), the session contains both question and answer."""
    system, mock_generate = rag
    mock_generate.return_value = "Supervised learning uses labelled examples."
    sid = system.session_manager.create_session()
    system.query("What is supervised learning?", session_id=sid)
    history = system.session_manager.get_conversation_history(sid)
    assert history is not None
    assert "supervised learning" in history.lower()
    assert "labelled examples" in history.lower()


def test_no_session_no_history(rag):
    """query(session_id=None) does not create or modify any session."""
    system, _ = rag
    system.query("What is overfitting?", session_id=None)
    assert system.session_manager.sessions == {}
