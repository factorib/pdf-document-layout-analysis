#!/bin/bash

# A10G GPU-Optimized PDF Layout Analysis Server Startup Script
# This script starts the production-ready server with full GPU acceleration

set -e  # Exit on any error

echo "🚀 Starting A10G GPU-Optimized PDF Layout Analysis Server"
echo "============================================================"

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

# Check if models exist
if [ ! -d "models" ]; then
    echo "❌ Error: Models directory not found"
    echo "   Please download the models first"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    echo "✅ GPU detected"
else
    echo "⚠️  Warning: nvidia-smi not found, GPU may not be available"
fi

# Activate detectron2 environment
echo "🐍 Activating detectron2 environment..."
source /home/ubuntu/envs/d2/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"
python3 -c "import fastapi; print('FastAPI: ✅')"
echo "✅ All dependencies found"

# Stop any existing server on port 80
echo "🛑 Stopping any existing server on port 80..."
sudo fuser -k 80/tcp 2>/dev/null || echo "   No existing server found"
sleep 2

# Create logs directory
mkdir -p logs

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down server..."
    sudo fuser -k 80/tcp 2>/dev/null || true
    echo "✅ Server stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the server
echo "🌐 Starting server on port 80..."
echo "📋 Server will be available at:"
echo "   - Local: http://localhost/"
echo "   - Network: http://$(hostname -I | awk '{print $1}')/"
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
echo "📝 Logs are saved to: logs/server_$(date +%Y%m%d_%H%M%S).log"
echo "🔧 Press Ctrl+C to stop the server"
echo ""
echo "Starting server..."

# Start server with logging
LOG_FILE="logs/server_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Server output will be logged to: $LOG_FILE"
echo "🔍 You can monitor logs in real-time with: tail -f $LOG_FILE"
echo ""

# Start server with both console output and logging
python3 run_a10g_server.py 2>&1 | tee "$LOG_FILE"