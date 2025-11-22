#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

ensure_compose() {
  if command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo "docker compose"
  fi
}

start_backend() {
  cd "$ROOT_DIR"
  uvicorn src.api.main:app --reload --port 8000
}

stop_backend() {
  # Match current uvicorn target
  pkill -f "uvicorn src.api.main:app" 2>/dev/null || true
}

start_frontend() {
  cd "$ROOT_DIR/ui"
  npm run dev
}

stop_frontend() {
  pkill -f "next dev" 2>/dev/null || \
  pkill -f "next-server" 2>/dev/null || true
}

dev_stack() {
  cd "$ROOT_DIR"
  echo "[dev] Starting backend on http://127.0.0.1:8000"
  uvicorn src.api.main:app --reload --port 8000 &
  BACKEND_PID=$!

  echo "[dev] Starting Next.js UI on http://localhost:3000"
  cd "$ROOT_DIR/ui"
  npm run dev &
  FRONTEND_PID=$!

  cleanup() {
    echo
    echo "[dev] Stopping dev stack..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  }

  trap cleanup INT TERM
  wait "$BACKEND_PID" "$FRONTEND_PID"
}

stack_up() {
  cd "$ROOT_DIR"
  COMPOSE_CMD="$(ensure_compose)"
  $COMPOSE_CMD up -d
}

stack_down() {
  cd "$ROOT_DIR"
  COMPOSE_CMD="$(ensure_compose)"
  $COMPOSE_CMD down
}

case "$1" in
  dev)
    dev_stack
    ;;
  b)
    case "$2" in
      start) start_backend ;;
      stop)  stop_backend ;;
      *)
        echo "Usage: ./run.sh b [start|stop]"
        exit 1
        ;;
    esac
    ;;
  f)
    case "$2" in
      start) start_frontend ;;
      stop)  stop_frontend ;;
      *)
        echo "Usage: ./run.sh f [start|stop]"
        exit 1
        ;;
    esac
    ;;
  stack)
    case "$2" in
      up)   stack_up ;;
      down) stack_down ;;
      *)
        echo "Usage: ./run.sh stack [up|down]"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Usage:"
    echo "  ./run.sh dev           # Start backend + frontend (local dev)"
    echo "  ./run.sh b start|stop  # Start/stop backend (uvicorn)"
    echo "  ./run.sh f start|stop  # Start/stop frontend (Next.js dev)"
    echo "  ./run.sh stack up|down # Start/stop Docker stack (API + Qdrant)"
    exit 1
    ;;
esac
