import sys
import os
from unittest.mock import MagicMock, patch

import chromadb
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_rag_system():
    """Preconfigured RAGSystem mock for API endpoint tests."""
    mock = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-id"
    mock.query.return_value = ("Test answer", ["source1", "source2"])
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "FastAPI Tutorial"],
    }
    return mock


@pytest.fixture(scope="session")
def app_module():
    """
    Import the FastAPI app with external dependencies neutralized.

    Patches are active only during the import itself to prevent:
    - StaticFiles(directory="../frontend") from failing (frontend dir absent in tests)
    - RAGSystem(config) from connecting to a real ChromaDB instance
    After import the module is cached; patches are released before yielding so they
    don't interfere with other fixtures.
    """
    import fastapi.staticfiles

    class _MockStaticFiles:
        """ASGI-compatible stub that skips the frontend directory check."""

        def __init__(self, *args, **kwargs):
            pass

        async def __call__(self, scope, receive, send):
            from starlette.responses import Response

            await Response("Not Found", status_code=404)(scope, receive, send)

    with patch.object(fastapi.staticfiles, "StaticFiles", _MockStaticFiles), \
         patch("vector_store.chromadb.PersistentClient",
               return_value=chromadb.EphemeralClient()), \
         patch("ai_generator.anthropic.Anthropic"):
        sys.modules.pop("app", None)
        import app as _app  # noqa: PLC0415

    # Patches no longer needed — module is cached, mounts/clients already initialised.
    yield _app

    sys.modules.pop("app", None)


@pytest.fixture
def client(app_module, mock_rag_system):
    """FastAPI TestClient backed by a mock RAGSystem."""
    from fastapi.testclient import TestClient

    original = app_module.rag_system
    app_module.rag_system = mock_rag_system
    with TestClient(app_module.app, raise_server_exceptions=False) as c:
        yield c
    app_module.rag_system = original
