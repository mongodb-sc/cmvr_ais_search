.PHONY: help setup install dev backend frontend stop clean verify start-bg

VENV_PATH := .venv
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip
BACKEND_PORT := 7860
FRONTEND_PORT := 3000

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║        CMVR/AIS Agentic Test Finder - Make Commands            ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Quick Start:"
	@echo "  make dev              Run both backend & frontend in foreground"
	@echo "  make start-bg         Run both services in background"
	@echo ""
	@echo "Development:"
	@echo "  make backend          Start FastAPI backend only (with reload)"
	@echo "  make frontend         Start Next.js frontend only"
	@echo ""
	@echo "Management:"
	@echo "  make setup            Setup environment & dependencies"
	@echo "  make install          Install Python dependencies"
	@echo "  make stop             Stop all background services"
	@echo "  make verify           Check if services are running"
	@echo "  make clean            Remove temporary files"
	@echo ""
	@echo "Service URLs:"
	@echo "  🎨 Frontend:  http://localhost:$(FRONTEND_PORT)"
	@echo "  📦 Backend:   http://localhost:$(BACKEND_PORT)"
	@echo "  🏥 Health:    http://localhost:$(BACKEND_PORT)/api/health"
	@echo ""

# Setup virtual environment and install dependencies
setup: install
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  make dev        # Run in foreground (recommended for development)"
	@echo "  make start-bg   # Run in background"

# Install Python dependencies
install:
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "📦 Creating virtual environment..."; \
		python3 -m venv $(VENV_PATH); \
	fi
	@echo "📦 Installing Python dependencies..."
	@$(PIP) install -q -r requirements.txt 2>/dev/null || $(PIP) install -r requirements.txt
	@echo "✓ Python dependencies installed"
	@echo ""
	@echo "📦 Installing Node dependencies..."
	@cd cmvr_agentic_ai/web && npm install -q 2>/dev/null || npm install
	@echo "✓ Node dependencies installed"

# Start both backend and frontend in foreground
dev: install
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║          🚀 Starting CMVR/AIS Application (DEV MODE)           ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Starting services..."
	@echo "  📦 Backend (FastAPI):  http://localhost:$(BACKEND_PORT)"
	@echo "  🎨 Frontend (Next.js): http://localhost:$(FRONTEND_PORT)"
	@echo ""
	@echo "Logs will appear below. Press Ctrl+C to stop all services."
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@$(PYTHON) -m uvicorn cmvr_agentic_ai.api:app --host 127.0.0.1 --port $(BACKEND_PORT) &
	@BACKEND_PID=$$!; \
	sleep 3; \
	cd cmvr_agentic_ai/web && npm run dev & \
	FRONTEND_PID=$$!; \
	trap "kill $$BACKEND_PID $$FRONTEND_PID 2>/dev/null" EXIT; \
	wait

# Start backend and frontend in background
start-bg: install
	@echo "🚀 Starting services in background..."
	@nohup $(PYTHON) -m uvicorn cmvr_agentic_ai.api:app --host 127.0.0.1 --port $(BACKEND_PORT) > /tmp/cmvr_backend.log 2>&1 &
	@sleep 3
	@echo "✓ Backend started (PID: $$!) - log: /tmp/cmvr_backend.log"
	@cd cmvr_agentic_ai/web && nohup npm run dev > /tmp/cmvr_frontend.log 2>&1 &
	@echo "✓ Frontend started (PID: $$!) - log: /tmp/cmvr_frontend.log"
	@echo ""
	@echo "🎉 Application is running!"
	@echo "   📍 Frontend: http://localhost:$(FRONTEND_PORT)"
	@echo "   📍 Backend:  http://localhost:$(BACKEND_PORT)"
	@echo ""
	@echo "To stop: make stop"
	@echo "To view logs: tail -f /tmp/cmvr_backend.log"

# Start backend only with reload
backend: install
	@echo "📦 Starting Backend (FastAPI) on http://localhost:$(BACKEND_PORT)..."
	@echo "Auto-reload enabled - changes to Python files will restart the server"
	@echo ""
	@$(PYTHON) -m uvicorn cmvr_agentic_ai.api:app --host 127.0.0.1 --port $(BACKEND_PORT) --reload

# Start frontend only
frontend:
	@echo "🎨 Starting Frontend (Next.js) on http://localhost:$(FRONTEND_PORT)..."
	@echo "Auto-reload enabled - changes to code will be reflected immediately"
	@echo ""
	@cd cmvr_agentic_ai/web && npm run dev

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	@pkill -f "uvicorn.*api:app" 2>/dev/null || true
	@pkill -f "next dev" 2>/dev/null || true
	@pkill -f "npm run dev" 2>/dev/null || true
	@sleep 1
	@echo "✓ All services stopped"

# Verify services are running
verify:
	@echo "🔍 Checking services..."
	@echo ""
	@if curl -s http://localhost:$(BACKEND_PORT)/api/health > /dev/null 2>&1; then \
		echo "✓ Backend is running on http://localhost:$(BACKEND_PORT)"; \
	else \
		echo "✗ Backend is NOT running"; \
	fi
	@if curl -s http://localhost:$(FRONTEND_PORT) > /dev/null 2>&1; then \
		echo "✓ Frontend is running on http://localhost:$(FRONTEND_PORT)"; \
	else \
		echo "✗ Frontend is NOT running"; \
	fi
	@echo ""

# Clean up temporary files
clean: stop
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache
	@cd cmvr_agentic_ai/web && rm -rf .next out dist
	@rm -f /tmp/cmvr_*.log
	@echo "✓ Cleanup complete"

# Default target
.DEFAULT_GOAL := help
