.PHONY: format check test

format:
	uv run black backend/

check:
	uv run black --check backend/

test:
	uv run pytest backend/tests/
