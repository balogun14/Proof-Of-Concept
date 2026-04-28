.PHONY: install test

# make help
help:
	@echo "Usage:"
	@echo "  make install                - Create env and install for local dev"
	@echo "  make train                  - Start Model Training"

# make install
install:
	uv sync --extra dev
	git lfs pull
	uv run pre-commit install

# make test
test:
	cd tests && pytest -v && cd ..

train:
	uv run python -m poc.train


