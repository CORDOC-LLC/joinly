#!/bin/bash
# Quick script to run Kurt with Gemini Live in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    if [ -f .env ]; then
        echo -e "${GREEN}Loading GOOGLE_API_KEY from .env file${NC}"
        export $(grep GOOGLE_API_KEY .env | xargs)
    fi

    if [ -z "$GOOGLE_API_KEY" ]; then
        echo -e "${RED}ERROR: GOOGLE_API_KEY not set${NC}"
        echo ""
        echo "Please set your Google API key:"
        echo "  export GOOGLE_API_KEY='your-api-key'"
        echo ""
        echo "Or create a .env file:"
        echo "  echo 'GOOGLE_API_KEY=your-api-key' > .env"
        echo ""
        echo "Get your API key from: https://aistudio.google.com/"
        exit 1
    fi
fi

# Check if meeting URL provided
if [ -z "$1" ]; then
    echo -e "${RED}ERROR: Meeting URL required${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <meeting-url> [agenda-file]"
    echo ""
    echo "Examples:"
    echo "  $0 https://meet.google.com/abc-defg-hij"
    echo "  $0 https://meet.google.com/abc-defg-hij examples/kurt_example_agenda.txt"
    exit 1
fi

MEETING_URL="$1"
AGENDA_FILE="$2"

# Image name
IMAGE_NAME="joinly-gemini:latest"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo -e "${YELLOW}Image $IMAGE_NAME not found. Building with BuildKit...${NC}"
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t "$IMAGE_NAME" .
else
    echo -e "${GREEN}Using existing image: $IMAGE_NAME${NC}"
fi

# Build docker run command
CMD="docker run --rm"
CMD="$CMD --env GOOGLE_API_KEY=$GOOGLE_API_KEY"
CMD="$CMD --shm-size=2gb"

# Add agenda file if provided
if [ -n "$AGENDA_FILE" ]; then
    if [ ! -f "$AGENDA_FILE" ]; then
        echo -e "${RED}ERROR: Agenda file not found: $AGENDA_FILE${NC}"
        exit 1
    fi
    CMD="$CMD -v $(pwd)/$AGENDA_FILE:/app/agenda.txt:ro"
    AGENDA_ARG="--agenda /app/agenda.txt"
    echo -e "${GREEN}Using agenda file: $AGENDA_FILE${NC}"
fi

# Add the image and command
CMD="$CMD $IMAGE_NAME"
CMD="$CMD examples/kurt_gemini.py"
if [ -n "$AGENDA_ARG" ]; then
    CMD="$CMD $AGENDA_ARG"
fi
CMD="$CMD \"$MEETING_URL\""

# Show what we're running
echo -e "${GREEN}Starting Kurt in meeting: $MEETING_URL${NC}"
echo ""

# Run it
eval $CMD
