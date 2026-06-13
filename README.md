# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Features

- **Course outline queries** — ask "what does course X cover?" to get a structured overview of all lessons and links, retrieved directly from the course catalog.
- **Semantic content search** — ask specific topic questions and get answers drawn from the relevant course chunks.
- **Multi-step reasoning** — Claude can chain up to 2 tool calls per query (configurable via `MAX_TOOL_ROUNDS` in `backend/config.py`), e.g. looking up a course outline and then searching within a specific lesson it reveals.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Running the Tests

```bash
# Basic run
uv run pytest backend/tests/

# Verbose — prints each test name and pass/fail status as tests execute
uv run pytest backend/tests/ -v
```

