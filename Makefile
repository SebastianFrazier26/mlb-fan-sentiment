.PHONY: setup test lint check docker-build docker-run

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

check: lint test

docker-build:
	docker build -t fansent .

# usage: make docker-run ARGS="rosters --season 2025"
docker-run:
	docker run --rm -v $(PWD)/data:/app/data fansent $(ARGS)
