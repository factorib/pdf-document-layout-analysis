#!/bin/bash

# A10G GPU-Optimized PDF Layout Analysis Server - Foreground Mode
# This script starts the server in foreground with visible output

set -e  # Exit on any error

echo "🚀 Starting A10G GPU-Optimized PDF Layout Analysis Server (Foreground)"
echo "======================================================================"

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

# Stop any existing server on port 80
echo "🛑 Stopping any existing server on port 80..."
sudo fuser -k 80/tcp 2>/dev/null || echo "   No existing server found"
sleep 2

echo ""
echo "🌐 Starting server on port 80..."
echo "📋 Server will be available at:"
echo "   - Local: http://localhost/"
echo "   - Network: http://$(hostname -I | awk '{print $1}')/"
echo ""
echo "📊 Available endpoints:"
echo "   - POST /           # Full PDF analysis (JSON output)"  
echo "   - POST /visualize  # Visual PDF with annotations"
echo "   - GET  /info       # Server information"
echo "   - GET  /status     # Processing status"
echo ""
echo "💡 Usage examples:"
echo "   curl -X POST -F 'file=@input.pdf' -F 'fast=false' -F 'extraction_format=' http://localhost/"
echo "   curl -X POST -F 'file=@input.pdf' -F 'fast=false' http://localhost/visualize > analyzed.pdf"
echo ""
echo "🔧 Press Ctrl+C to stop the server"
echo "🔄 All output will be visible in this terminal"
echo ""

# Start server with full output visible
python3 run_a10g_server.py