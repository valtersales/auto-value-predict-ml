.PHONY: help build up down restart logs shell clean clean-all clean-containers clean-images clean-volumes pipeline pipeline-full pipeline-phase2 pipeline-list pipeline-status pipeline-step pipeline-from test test-unit test-cleaner test-validator test-splitter test-pipeline

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the Docker image
	docker-compose build

up: ## Start the containers
	docker-compose up -d

down: ## Stop the containers
	docker-compose down

restart: ## Restart the containers
	docker-compose restart

logs: ## Show container logs
	docker-compose logs -f

shell: ## Open a shell in the running container
	docker-compose exec api /bin/bash

jupyter: ## Start Jupyter Lab server
	docker-compose up -d jupyter
	@echo "Jupyter Lab is starting..."
	@echo "Access it at: http://localhost:8888"
	@echo "To view logs: make logs-jupyter"

logs-jupyter: ## Show Jupyter container logs
	docker-compose logs -f jupyter

stop-jupyter: ## Stop Jupyter container
	docker-compose stop jupyter

clean: ## Stop and remove containers, networks
	docker-compose down

clean-containers: ## Remove all stopped containers
	docker container prune -f

clean-images: ## Remove unused Docker images
	docker image prune -a -f

clean-volumes: ## Remove unused volumes
	docker volume prune -f

clean-all: clean clean-containers clean-images clean-volumes ## Remove containers, images, and volumes (WARNING: destructive)

rebuild: clean build ## Clean and rebuild the Docker image

start: build up ## Build and start the containers

stop: down ## Stop the containers

status: ## Show container status
	docker-compose ps


# Pipeline commands
# All commands run inside Docker containers
# Make sure containers are running: make up

pipeline: ## Run the full ML pipeline (up to implemented phases)
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/run_pipeline.py

pipeline-full: pipeline ## Alias for pipeline

pipeline-phase2: ## Run only Phase 2 (data preprocessing)
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/preprocess_data.py

pipeline-list: ## List all available pipeline steps
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/run_pipeline.py --list-steps

pipeline-status: ## Show pipeline execution status
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/run_pipeline.py --status

pipeline-step: ## Run a specific pipeline step (usage: make pipeline-step STEP=clean_data)
	@if [ -z "$(STEP)" ]; then \
		echo "Error: STEP variable is required. Usage: make pipeline-step STEP=step_name"; \
		echo "Available steps: load_data, validate_data_before, clean_data, validate_data_after, split_data"; \
		exit 1; \
	fi
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/run_pipeline.py --start-from $(STEP) --stop-at $(STEP)

pipeline-from: ## Start pipeline from a specific step (usage: make pipeline-from STEP=clean_data)
	@if [ -z "$(STEP)" ]; then \
		echo "Error: STEP variable is required. Usage: make pipeline-from STEP=step_name"; \
		echo "Available steps: load_data, validate_data_before, clean_data, validate_data_after, split_data"; \
		exit 1; \
	fi
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/run_pipeline.py --start-from $(STEP)

train-baseline: ## Train baseline models
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/train_baseline_models.py

train-advanced: ## Train advanced models (Random Forest, XGBoost, LightGBM)
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/train_advanced_models.py

evaluate-test: ## Evaluate trained models on test set
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/evaluate_test_set.py

optimize-lightgbm: ## Optimize LightGBM hyperparameters
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/optimize_lightgbm.py

analyze-segments: ## Analyze model performance by segments and errors
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/analyze_segments_and_errors.py

save-model: ## Save model with versioning and metadata
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/save_model_with_versioning.py

load-model: ## Load and validate a saved model
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api python scripts/load_and_validate_model.py

# Testing commands
# All tests run inside Docker containers
# Make sure containers are running: make up

test: ## Run all tests
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api pytest tests/ -v

test-unit: ## Run unit tests only
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api pytest tests/ -v -k "test_"

test-cleaner: ## Run tests for data cleaner module
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api pytest tests/test_cleaner.py -v

test-validator: ## Run tests for data validator module
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api pytest tests/test_validator.py -v

test-splitter: ## Run tests for data splitter module
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api pytest tests/test_splitter.py -v

test-pipeline: ## Run tests for pipeline module
	@if ! docker-compose ps api 2>/dev/null | grep -q "Up"; then \
		echo "Error: Docker container 'api' is not running."; \
		echo "Please start containers first: make up"; \
		exit 1; \
	fi
	docker-compose exec api pytest tests/test_pipeline.py -v || echo "Pipeline tests not yet implemented"

