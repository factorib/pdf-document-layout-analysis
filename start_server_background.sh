#!/bin/bash

# A10G GPU-Optimized PDF Layout Analysis Server Background Startup Script
# This script starts the server in the background for production deployment

set -e  # Exit on any error

echo "🚀 Starting A10G GPU-Optimized PDF Layout Analysis Server (Background Mode)"
echo "============================================================================"

# Check if we're in the right directory
if [ ! -f "run_a10g_server.py" ]; then
    echo "❌ Error: run_a10g_server.py not found"
    echo "   Please run this script from the pdf-document-layout-analysis directory"
    exit 1
fi

# Check if detectron2 environment exists
if [ ! -d "/home/ubuntu/envs/d2" ]; then
    echo "❌ Error: Detectron2 environment not found at /home/ubuntu/envs/d2"
    echo "   Please run the installation script first"
    exit 1
fi

# Create logs and pid directories
mkdir -p logs
mkdir -p pids

# Stop any existing server
echo "🛑 Stopping any existing server..."
if [ -f "pids/server.pid" ]; then
    SERVER_PID=$(cat pids/server.pid 2>/dev/null || echo "")
    if [ ! -z "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "   Stopping existing server (PID: $SERVER_PID)"
        kill "$SERVER_PID"
        sleep 3
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "   Force killing server..."
            kill -9 "$SERVER_PID"
        fi
    fi
fi

# Kill any process on port 80
sudo fuser -k 80/tcp 2>/dev/null || echo "   No process found on port 80"
sleep 2

# Activate detectron2 environment and start server in background
echo "🌐 Starting server in background mode..."
LOG_FILE="logs/server_$(date +%Y%m%d_%H%M%S).log"

# Start server in background
source /home/ubuntu/envs/d2/bin/activate && \
nohup python3 run_a10g_server.py > "$LOG_FILE" 2>&1 &

# Save PID
SERVER_PID=$!
echo $SERVER_PID > pids/server.pid

echo "✅ Server started successfully!"
echo "📋 Server details:"
echo "   - PID: $SERVER_PID"
echo "   - Log file: $LOG_FILE"
echo "   - URL: http://localhost/"
echo "   - Network: http://$(hostname -I | awk '{print $1}')/"
echo ""

# Wait for server to initialize
echo "⏳ Waiting for server to initialize..."
sleep 15

# Check if server is responding
if curl -s http://localhost/ > /dev/null 2>&1; then
    echo "✅ Server is responding to requests"
    echo ""
    echo "📊 Available endpoints:"
    echo "   - POST /           # Full PDF analysis (JSON output)"  
    echo "   - POST /visualize  # Visual PDF with annotations"
    echo "   - GET  /info       # Server information"
    echo ""
    echo "💡 Usage examples:"
    echo "   curl -X POST -F 'file=@input.pdf' -F 'fast=false' -F 'extraction_format=' http://localhost/"
    echo "   curl -X POST -F 'file=@input.pdf' -F 'fast=false' http://localhost/visualize > analyzed.pdf"
    echo ""
    echo "🔧 To stop the server: ./stop_server.sh"
    echo "📝 To view logs: tail -f $LOG_FILE"
else
    echo "❌ Server failed to start or is not responding"
    echo "📝 Check the log file: $LOG_FILE"
    exit 1
fi