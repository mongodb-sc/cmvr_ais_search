# Running CMVR/AIS Agentic Test Finder

## Quick Start

The easiest way to run the entire application is using the Makefile:

```bash
# Start both backend and frontend in foreground (recommended for development)
make dev

# Or start in background mode
make start-bg
```

## Individual Commands

### First Time Setup
```bash
# Install all dependencies (Python and Node)
make setup
```

### Development Mode (Recommended)
Run both services with live reload:
```bash
make dev
```
This will:
- Start the FastAPI backend on **http://localhost:7860**
- Start the Next.js frontend on **http://localhost:3000**
- Display logs in your terminal
- Auto-reload on code changes

### Running Services Separately
```bash
# Terminal 1: Backend only (with auto-reload)
make backend

# Terminal 2: Frontend only (with auto-reload)
make frontend
```

### Background Mode
Run services without blocking terminal output:
```bash
make start-bg
```
View logs:
```bash
tail -f /tmp/cmvr_backend.log   # Backend logs
tail -f /tmp/cmvr_frontend.log  # Frontend logs
```

## Service URLs

Once running:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:7860
- **Health Check:** http://localhost:7860/api/health

## Stopping Services

```bash
# Stop all running services
make stop
```

## Verification

Check if services are running:
```bash
make verify
```

## Cleanup

Remove temporary files and build artifacts:
```bash
make clean
```

## Troubleshooting

### Port Already in Use
If port 7860 or 3000 is already in use:
```bash
# Stop existing services first
make stop

# Or manually kill processes:
pkill -f "uvicorn"
pkill -f "next dev"
pkill -f "npm"
```

### Dependencies Not Installed
Reinstall dependencies:
```bash
make install
```

### View All Available Commands
```bash
make help
```

## Architecture

```
┌─────────────────────────────────────────┐
│      Next.js Frontend (Port 3000)       │
│    • React components                   │
│    • API client integration             │
└────────────────┬────────────────────────┘
                 │
                 │ HTTP Requests
                 ▼
┌─────────────────────────────────────────┐
│     FastAPI Backend (Port 7860)         │
│    • /api/chat - Stream responses       │
│    • /api/history - Query history       │
│    • /api/health - Health check         │
└────────────────┬────────────────────────┘
                 │
                 │ MongoDB Queries
                 ▼
         MongoDB Database
      (CMVR/AIS Rules)
```

## Environment Variables

The frontend connects to the backend via `NEXT_PUBLIC_API_URL` environment variable.

This is already configured in:
- `cmvr_agentic_ai/web/.env.local` (localhost:7860)

For production deployment, update this value accordingly.
