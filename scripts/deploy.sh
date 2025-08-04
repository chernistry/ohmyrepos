#!/bin/bash
# Production deployment script for Oh My Repos

set -euo pipefail

# Configuration
DOCKER_IMAGE="ohmyrepos:latest"
CONTAINER_NAME="ohmyrepos-prod"
ENV_FILE="${ENV_FILE:-.env}"
PORT="${PORT:-8501}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || error "Docker is not installed"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is not installed"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file $ENV_FILE not found. Please copy .env-example to .env and configure it."
    fi
    
    log "Prerequisites check passed"
}

# Validate environment
validate_environment() {
    log "Validating environment configuration..."
    
    # Required environment variables
    local required_vars=(
        "GITHUB_USERNAME"
        "GITHUB_TOKEN"
        "EMBEDDING_MODEL_API_KEY"
        "QDRANT_URL"
    )
    
    local missing_vars=()
    
    # Source the environment file
    set -a
    source "$ENV_FILE"
    set +a
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
    fi
    
    log "Environment validation passed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    docker build --target production -t "$DOCKER_IMAGE" .
    
    log "Docker image built successfully"
}

# Stop existing container
stop_existing() {
    log "Stopping existing container if running..."
    
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
        log "Existing container stopped and removed"
    else
        log "No existing container found"
    fi
}

# Run container
run_container() {
    log "Starting new container..."
    
    docker run -d \
        --name "$CONTAINER_NAME" \
        --env-file "$ENV_FILE" \
        -p "$PORT:8501" \
        --restart unless-stopped \
        --health-cmd="curl -f http://localhost:8501/health || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        "$DOCKER_IMAGE"
    
    log "Container started successfully"
}

# Wait for health check
wait_for_health() {
    log "Waiting for application to be healthy..."
    
    local timeout=60
    local count=0
    
    while [[ $count -lt $timeout ]]; do
        if docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "healthy"; then
            log "Application is healthy"
            return 0
        fi
        
        if [[ $count -eq 0 ]]; then
            echo -n "Waiting"
        fi
        echo -n "."
        sleep 1
        ((count++))
    done
    
    echo ""
    error "Application failed to become healthy within $timeout seconds"
}

# Show status
show_status() {
    log "Deployment status:"
    echo ""
    echo "Container Status:"
    docker ps -f name="$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "Health Status:"
    docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "No health check available"
    echo ""
    echo "Application URL: http://localhost:$PORT"
    echo ""
    echo "View logs: docker logs $CONTAINER_NAME"
    echo "Stop application: docker stop $CONTAINER_NAME"
}

# Main deployment function
main() {
    log "Starting Oh My Repos deployment..."
    
    check_prerequisites
    validate_environment
    build_image
    stop_existing
    run_container
    wait_for_health
    show_status
    
    log "Deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log "Stopping Oh My Repos..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || warn "Container not running"
        docker rm "$CONTAINER_NAME" 2>/dev/null || warn "Container not found"
        log "Stopped successfully"
        ;;
    "logs")
        docker logs -f "$CONTAINER_NAME"
        ;;
    "status")
        show_status
        ;;
    "restart")
        $0 stop
        $0 deploy
        ;;
    *)
        echo "Usage: $0 {deploy|stop|logs|status|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the application (default)"
        echo "  stop     - Stop the application"
        echo "  logs     - View application logs"
        echo "  status   - Show application status"
        echo "  restart  - Restart the application"
        exit 1
        ;;
esac