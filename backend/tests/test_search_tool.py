import pytest
import chromadb
from unittest.mock import MagicMock, patch

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool


@pytest.fixture
def mock_store():
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.return_value = None
    return store


@pytest.fixture
def real_store_good(tmp_path):
    """Real VectorStore backed by in-memory ChromaDB, max_results=5 (correct value)."""
    with patch("vector_store.chromadb.PersistentClient") as mock_pc:
        mock_pc.return_value = chromadb.EphemeralClient()
        store = VectorStore(
            chroma_path=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        yield store


@pytest.fixture
def real_store_zero(tmp_path):
    """Real VectorStore with max_results=0 (the bug value), one doc pre-loaded."""
    with patch("vector_store.chromadb.PersistentClient") as mock_pc:
        mock_pc.return_value = chromadb.EphemeralClient()
        store = VectorStore(
            chroma_path=str(tmp_path),
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,
        )
        store.course_content.add(
            documents=["Machine learning is a subset of artificial intelligence."],
            metadatas=[
                {"course_title": "ML Intro", "lesson_number": 1, "chunk_index": 0}
            ],
            ids=["chunk_000"],
        )
        yield store


def test_happy_path(mock_store):
    """execute() returns formatted string with course title and document content."""
    mock_store.search.return_value = SearchResults(
        documents=["Supervised learning uses labelled training data."],
        metadata=[{"course_title": "ML Intro", "lesson_number": 1}],
        distances=[0.2],
    )
    tool = CourseSearchTool(mock_store)
    result = tool.execute(query="What is supervised learning?")
    assert isinstance(result, str)
    assert "ML Intro" in result
    assert "Supervised learning" in result


def test_error_from_store_returns_string_not_raises(mock_store):
    """
    When VectorStore.search() returns SearchResults with an error, execute()
    returns the error string — it does NOT raise. This is the primary bug path:
    MAX_RESULTS=0 causes a ChromaDB error, which is absorbed here as a string
    returned to the AI rather than crashing the server.
    """
    mock_store.search.return_value = SearchResults.empty(
        "Search error: Number of requested results 0, cannot be negative, or zero. in query."
    )
    tool = CourseSearchTool(mock_store)
    result = tool.execute(query="What is machine learning?")
    assert isinstance(result, str)
    assert "Search error" in result


def test_empty_results_returns_no_content_message(mock_store):
    """When search returns no documents (and no error), execute() says 'No relevant content found'."""
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    tool = CourseSearchTool(mock_store)
    result = tool.execute(query="What is quantum computing?")
    assert "No relevant content found" in result


def test_passes_course_name_to_store(mock_store):
    """course_name argument is forwarded unchanged to VectorStore.search()."""
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    tool = CourseSearchTool(mock_store)
    tool.execute(query="neural networks", course_name="Deep Learning 101")
    mock_store.search.assert_called_once_with(
        query="neural networks",
        course_name="Deep Learning 101",
        lesson_number=None,
    )


def test_passes_lesson_number_to_store(mock_store):
    """lesson_number argument is forwarded unchanged to VectorStore.search()."""
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    tool = CourseSearchTool(mock_store)
    tool.execute(query="backpropagation", lesson_number=3)
    mock_store.search.assert_called_once_with(
        query="backpropagation",
        course_name=None,
        lesson_number=3,
    )


def test_max_results_zero_falls_back_to_five(real_store_zero):
    """
    Regression test for the MAX_RESULTS=0 bug.

    After Fix 2 (guard `if search_limit < 1: search_limit = 5`), VectorStore.search()
    never passes 0 to ChromaDB. Instead it falls back to 5, so the search
    succeeds and execute() returns content — not a 'Search error' string.
    """
    tool = CourseSearchTool(real_store_zero)
    result = tool.execute(query="machine learning")
    assert isinstance(result, str), "execute() must return a string, not raise"
    assert (
        "Search error" not in result
    ), f"Expected no 'Search error' after fix, got: {result!r}"


def test_max_results_five_real_chromadb(real_store_good):
    """After applying Fix 1 (MAX_RESULTS=5), real ChromaDB returns actual content."""
    real_store_good.course_content.add(
        documents=["Gradient descent minimises the loss function iteratively."],
        metadatas=[{"course_title": "ML Intro", "lesson_number": 2, "chunk_index": 0}],
        ids=["chunk_001"],
    )
    tool = CourseSearchTool(real_store_good)
    result = tool.execute(query="gradient descent")
    assert isinstance(result, str)
    assert "Search error" not in result
    assert "Gradient descent" in result
