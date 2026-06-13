# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# From repo root (requires Git Bash on Windows)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is served at `http://localhost:8000`. The API docs are at `http://localhost:8000/docs`.

Install dependencies with `uv sync`. Add new packages with `uv add <package>`. Run Python scripts with `uv run python <file>`. Requires an `ANTHROPIC_API_KEY` in `.env` at the repo root.

## Architecture

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot:

- **Backend**: FastAPI app (`backend/`) serving both the REST API and the static frontend
- **Frontend**: Vanilla JS/HTML/CSS (`frontend/`) — no build step
- **Vector DB**: ChromaDB persisted at `backend/chroma_db/` (relative to where uvicorn runs)
- **LLM**: Anthropic Claude (model configured in `backend/config.py`)
- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers`

### Request flow

```
POST /api/query
  → RAGSystem.query()
    → AIGenerator.generate_response()  (Claude API with tool_choice=auto)
      → Claude calls get_course_outline or search_course_content tool
        → ToolManager.execute_tool()
          → CourseOutlineTool.execute()  (outline/overview queries)
              → VectorStore.get_course_outline()  (course_catalog lookup)
          → CourseSearchTool.execute()  (specific content queries)
              → VectorStore.search()  (ChromaDB semantic search)
      ↑ tool calls repeat up to MAX_TOOL_ROUNDS (config.py, default 2)
      → Claude receives all tool results, generates final answer
  → SessionManager records exchange
  → sources returned from ToolManager.get_last_sources()
```

### Key components

| File | Role |
|------|------|
| `backend/app.py` | FastAPI entrypoint; loads docs from `../docs` on startup |
| `backend/rag_system.py` | Orchestrator — wires all components together |
| `backend/ai_generator.py` | Claude API calls; handles sequential tool-use loop (up to `MAX_TOOL_ROUNDS` per query) |
| `backend/vector_store.py` | ChromaDB wrapper; two collections: `course_catalog` and `course_content` |
| `backend/document_processor.py` | Parses `.txt`/`.pdf`/`.docx` course files → `Course` + `CourseChunk` objects |
| `backend/search_tools.py` | `Tool` ABC + `CourseOutlineTool` + `CourseSearchTool` + `ToolManager` |
| `backend/session_manager.py` | In-memory conversation history (keyed by session ID) |
| `backend/models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |
| `backend/config.py` | All tuneable constants (chunk size, model name, history length, etc.) |

### Course document format

Text files in `docs/` must follow this structure:

```
Course Title: <title>          ← used as the unique ID in ChromaDB
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>             ← optional, must immediately follow lesson header
<lesson content...>

Lesson 1: <next lesson title>
<content...>
```

The course title is the primary key — re-uploading a file with the same title is skipped at startup.

### ChromaDB collections

**`course_catalog`** — course titles for name resolution
- Metadata per course: `title`, `instructor`, `course_link`, `lesson_count`, `lessons_json`
- `lessons_json` is a serialized list of `{lesson_number, lesson_title, lesson_link}`

**`course_content`** — text chunks for semantic search
- Metadata per chunk: `course_title`, `lesson_number`, `chunk_index`
- IDs are `{course_title}_{chunk_index}`

Course name resolution in `VectorStore.search()` uses semantic search against `course_catalog` before filtering `course_content`, so partial/fuzzy course names work.

### Registered Claude tools

| Tool name | Class | When Claude uses it |
|-----------|-------|---------------------|
| `get_course_outline` | `CourseOutlineTool` | Outline, overview, or "what does course X cover?" queries — reads full lesson list from `course_catalog` |
| `search_course_content` | `CourseSearchTool` | Specific topic or content questions — semantic chunk search against `course_content` |

Claude may chain these tools across up to `MAX_TOOL_ROUNDS` rounds (default **2**, set in `config.py`). After the budget is exhausted it is force-stopped and must produce a final text answer. Hard tool failures are surfaced as `is_error` tool_result blocks so Claude can still give a best-effort response.

### Tests

```bash
uv run pytest backend/tests/
```

| File | Coverage |
|------|----------|
| `backend/tests/test_search_tool.py` | `CourseSearchTool` and `CourseOutlineTool` unit tests |
| `backend/tests/test_ai_generator.py` | `AIGenerator` tool-call wiring, multi-round logic, error handling |
| `backend/tests/test_rag_system.py` | `RAGSystem.query()` end-to-end flow |
