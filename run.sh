#!/bin/bash

case "$1 $2" in
  "b start")
    cd "$(dirname "$0")"
    uvicorn src.api.main:app --reload --port 8000
    ;;
  "b stop")
    pkill -f "uvicorn src.main:app"
    ;;
  "f start")
    cd "$(dirname "$0")/ui"
    npm run dev
    ;;
  "f stop")
    pkill -f "next-server"
    ;;
  *)
    echo "Usage: ./run.sh [b|f] [start|stop]"
    echo "  b start - Start backend"
    echo "  b stop  - Stop backend"
    echo "  f start - Start frontend"
    echo "  f stop  - Stop frontend"
    exit 1
    ;;
esac
