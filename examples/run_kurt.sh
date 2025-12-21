#!/bin/bash

# run_kurt.sh
# Convenience script to run Kurt with common configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MCP_URL="http://localhost:8000/mcp/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENDA_FILE="${SCRIPT_DIR}/kurt_example_agenda.txt"
MODEL_NAME="${JOINLY_MODEL_NAME:-gpt-4o}"
MODEL_PROVIDER="${JOINLY_MODEL_PROVIDER:-openai}"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if joinly server is running
check_server() {
    print_info "Checking if Joinly MCP server is running at ${MCP_URL}..."
    if curl -s --max-time 3 "http://localhost:8000/health" > /dev/null 2>&1; then
        print_info "Joinly server is running ✓"
        return 0
    else
        print_warning "Joinly server not detected at http://localhost:8000"
        return 1
    fi
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] MEETING_URL

Run Kurt, the meeting facilitation agent.

Arguments:
    MEETING_URL         The URL of the meeting to join (required)

Options:
    -a, --agenda FILE   Path to agenda file (default: kurt_example_agenda.txt)
    -m, --model NAME    LLM model name (default: gpt-4o or JOINLY_MODEL_NAME env)
    -p, --provider PROV LLM provider (default: openai or JOINLY_MODEL_PROVIDER env)
    -c, --config FILE   Path to MCP servers config JSON
    -u, --mcp-url URL   Joinly MCP server URL (default: http://localhost:8000/mcp/)
    -h, --help          Display this help message

Examples:
    # Basic usage with example agenda
    $0 https://meet.google.com/abc-defg-hij

    # With custom agenda
    $0 --agenda my_agenda.txt https://meet.google.com/abc-defg-hij

    # Using Claude instead of GPT
    $0 --model claude-3-5-sonnet-latest --provider anthropic \\
       https://meet.google.com/abc-defg-hij

    # With additional MCP servers
    $0 --config mcp_config.json --agenda my_agenda.txt \\
       https://meet.google.com/abc-defg-hij

    # No agenda (Kurt will still facilitate)
    $0 --agenda "" https://meet.google.com/abc-defg-hij

Environment Variables:
    JOINLY_MODEL_NAME      Default LLM model name
    JOINLY_MODEL_PROVIDER  Default LLM provider
    OPENAI_API_KEY        API key for OpenAI
    ANTHROPIC_API_KEY     API key for Anthropic
    ELEVENLABS_API_KEY    API key for ElevenLabs TTS

EOF
    exit 0
}

# Parse command line arguments
CONFIG_FILE=""
USE_DEFAULT_AGENDA=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--agenda)
            AGENDA_FILE="$2"
            USE_DEFAULT_AGENDA=false
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -p|--provider)
            MODEL_PROVIDER="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -u|--mcp-url)
            MCP_URL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            print_error "Unknown option: $1"
            usage
            ;;
        *)
            MEETING_URL="$1"
            shift
            ;;
    esac
done

# Check if meeting URL is provided
if [ -z "$MEETING_URL" ]; then
    print_error "Meeting URL is required"
    usage
fi

# Validate meeting URL format
if [[ ! "$MEETING_URL" =~ ^https?:// ]]; then
    print_error "Invalid meeting URL format. Must start with http:// or https://"
    exit 1
fi

print_info "==================================================="
print_info "          Kurt - Meeting Facilitation Agent        "
print_info "==================================================="
echo ""

# Check if .env file exists
if [ ! -f "${SCRIPT_DIR}/../.env" ]; then
    print_warning "No .env file found at ${SCRIPT_DIR}/../.env"
    print_warning "Make sure you have set up your API keys as environment variables"
fi

# Check server
if ! check_server; then
    print_warning "You may need to start the Joinly server first:"
    print_warning "  docker run -p 8000:8000 --env-file .env ghcr.io/joinly-ai/joinly:latest"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the command
CMD="python ${SCRIPT_DIR}/kurt_agent.py"

# Add agenda if specified
if [ "$USE_DEFAULT_AGENDA" = true ] && [ -f "$AGENDA_FILE" ]; then
    print_info "Using agenda file: ${AGENDA_FILE}"
    CMD="${CMD} --agenda \"${AGENDA_FILE}\""
elif [ -n "$AGENDA_FILE" ] && [ "$AGENDA_FILE" != "" ]; then
    if [ -f "$AGENDA_FILE" ]; then
        print_info "Using custom agenda file: ${AGENDA_FILE}"
        CMD="${CMD} --agenda \"${AGENDA_FILE}\""
    else
        print_warning "Agenda file not found: ${AGENDA_FILE}"
        print_info "Running without agenda"
    fi
else
    print_info "Running without agenda"
fi

# Add model settings
print_info "Using LLM: ${MODEL_NAME} (${MODEL_PROVIDER})"
CMD="${CMD} --model-name \"${MODEL_NAME}\""
if [ -n "$MODEL_PROVIDER" ]; then
    CMD="${CMD} --model-provider \"${MODEL_PROVIDER}\""
fi

# Add MCP URL
CMD="${CMD} --mcp-url \"${MCP_URL}\""

# Add config if specified
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    print_info "Using MCP config: ${CONFIG_FILE}"
    CMD="${CMD} --config \"${CONFIG_FILE}\""
fi

# Add meeting URL
CMD="${CMD} \"${MEETING_URL}\""

print_info "Meeting URL: ${MEETING_URL}"
echo ""
print_info "Starting Kurt..."
echo ""

# Execute the command
eval $CMD
