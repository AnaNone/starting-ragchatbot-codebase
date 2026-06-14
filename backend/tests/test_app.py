"""Tests for the FastAPI API endpoints: /api/query, /api/courses, /."""


class TestQueryEndpoint:
    def test_returns_answer_and_sources(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["source1", "source2"]
        assert "session_id" in data

    def test_creates_session_when_none_provided(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "Hello"})
        assert response.status_code == 200
        mock_rag_system.session_manager.create_session.assert_called_once()
        assert response.json()["session_id"] == "test-session-id"

    def test_uses_provided_session_id(self, client, mock_rag_system):
        response = client.post(
            "/api/query", json={"query": "Hello", "session_id": "existing-session"}
        )
        assert response.status_code == 200
        mock_rag_system.session_manager.create_session.assert_not_called()
        assert response.json()["session_id"] == "existing-session"

    def test_passes_correct_args_to_rag(self, client, mock_rag_system):
        client.post(
            "/api/query", json={"query": "Tell me about FastAPI", "session_id": "sess-1"}
        )
        mock_rag_system.query.assert_called_once_with("Tell me about FastAPI", "sess-1")

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_backend_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB unavailable")
        response = client.post("/api/query", json={"query": "test"})
        assert response.status_code == 500
        assert "DB unavailable" in response.json()["detail"]


class TestCoursesEndpoint:
    def test_returns_course_stats(self, client):
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Python Basics", "FastAPI Tutorial"]

    def test_backend_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("broken")
        response = client.get("/api/courses")
        assert response.status_code == 500
        assert "broken" in response.json()["detail"]


class TestRootEndpoint:
    def test_root_does_not_return_server_error(self, client):
        """Static files are unavailable in the test env; expect 404, not 5xx."""
        response = client.get("/")
        assert response.status_code < 500
