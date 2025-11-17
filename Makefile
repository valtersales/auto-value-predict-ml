.PHONY: help build up down restart logs shell clean clean-all clean-containers clean-images clean-volumes

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

