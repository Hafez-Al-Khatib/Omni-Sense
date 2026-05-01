# ──────────────────────────────────────────────────────────────────────────────
# Omni-Sense — DevOps Makefile
# Capstone deployment commands
# ──────────────────────────────────────────────────────────────────────────────

.ONESHELL:
SHELL := /bin/bash
.PHONY: lint test build push deploy-local deploy-gcp smoke-test clean

# Configurable variables
REGISTRY ?= ghcr.io/omni-sense
TAG      ?= latest
GCP_PROJECT_ID ?=
GCP_REGION     ?= us-central1

PYTHON := python3
DOCKER_SERVICES := eep iep2 iep3 iep4 omni-platform

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:
	@echo "🔍 Running ruff linter..."
	ruff check . --output-format=github --ignore E501,E402 || true

# ── Test ──────────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Running all test suites..."
	cd eep && $(PYTHON) -m pytest tests/ -v --tb=short
	cd iep2 && $(PYTHON) -m pytest tests/ -v --tb=short
	cd iep4 && $(PYTHON) -m pytest tests/ -v --tb=short
	$(PYTHON) -m pytest omni/tests/ -v --tb=short -q

# ── Build ─────────────────────────────────────────────────────────────────────
build:
	@echo "🐳 Building all Docker images..."
	docker build -t $(REGISTRY)/omni-sense-eep:$(TAG) ./eep
	docker build -t $(REGISTRY)/omni-sense-iep2:$(TAG) -f iep2/Dockerfile .
	docker build -t $(REGISTRY)/omni-sense-iep3:$(TAG) -f iep3/Dockerfile .
	docker build -t $(REGISTRY)/omni-sense-iep4:$(TAG) -f iep4/Dockerfile .
	docker build -t $(REGISTRY)/omni-sense-platform:$(TAG) -f omni/Dockerfile .

# ── Push ──────────────────────────────────────────────────────────────────────
push: build
	@echo "📤 Pushing images to registry $(REGISTRY)..."
	@echo "Ensure you are logged in: docker login $(REGISTRY)"
	@for svc in eep iep2 iep3 iep4 platform; do \
		docker push $(REGISTRY)/omni-sense-$$svc:$(TAG); \
	done

# ── Local Deploy (Docker Compose) ─────────────────────────────────────────────
deploy-local:
	@echo "🚀 Starting local stack with Docker Compose..."
	docker compose up -d

# ── Local Deploy (Kubernetes via Helm, requires kind + helm) ──────────────────
deploy-k8s:
	@echo "☸️  Deploying to local Kubernetes cluster..."
	@echo "Ensure kind cluster exists: kind create cluster --name omni-sense"
	kubectl create namespace omni-sense --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install omni-sense ./k8s/helm/omni-sense \
		--namespace omni-sense \
		--wait --timeout 10m

# ── GCP Deploy ────────────────────────────────────────────────────────────────
deploy-gcp:
	@echo "🌩️  Deploying to GCP..."
	@if command -v terraform >/dev/null 2>&1; then \
		echo "Using Terraform..."; \
		cd infra/terraform && terraform init && terraform apply -auto-approve; \
	else \
		echo "Terraform not found; falling back to gcloud script..."; \
		bash scripts/deploy-gcp.sh; \
	fi

# ── Smoke Test ────────────────────────────────────────────────────────────────
smoke-test:
	@echo "🚬 Running smoke tests..."
	@echo "EEP health:"
	@curl -sf http://localhost:8000/health || echo "  EEP not reachable on localhost:8000"
	@echo "IEP2 health:"
	@curl -sf http://localhost:8002/health || echo "  IEP2 not reachable on localhost:8002"
	@echo "IEP3 health:"
	@curl -sf http://localhost:8003/health || echo "  IEP3 not reachable on localhost:8003"
	@echo "IEP4 health:"
	@curl -sf http://localhost:8004/health || echo "  IEP4 not reachable on localhost:8004"
	@echo "Prometheus:"
	@curl -sf http://localhost:9090/-/healthy || echo "  Prometheus not reachable on localhost:9090"
	@echo "Grafana:"
	@curl -sf http://localhost:3000/api/health || echo "  Grafana not reachable on localhost:3000"

# ── Destroy ───────────────────────────────────────────────────────────────────
destroy-gcp:
	@echo "🧨 Destroying GCP resources..."
	@if command -v terraform >/dev/null 2>&1; then \
		cd infra/terraform && terraform destroy -auto-approve; \
	else \
		bash scripts/destroy-gcp.sh; \
	fi

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning up..."
	docker compose down -v || true
	docker system prune -f || true
