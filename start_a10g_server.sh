#!/bin/bash

# A10G GPU-Optimized PDF Layout Analysis Server Startup Script
# This script starts the native optimized server with detectron2 support

echo "🚀 Starting A10G GPU-Optimized PDF Layout Analysis Server"

# Activate detectron2 environment with all dependencies
source /home/ubuntu/envs/d2/bin/activate

# Set Python path for the project
export PYTHONPATH="/home/ubuntu/pdf-document-layout-analysis/src:$PYTHONPATH"

# GPU optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Change to project directory
cd /home/ubuntu/pdf-document-layout-analysis

# Check GPU availability
echo "🔍 Checking GPU status..."
python3 -c "import torch; print(f'✅ GPU Available: {torch.cuda.is_available()}'); print(f'✅ GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"

# Check detectron2
echo "🔍 Checking detectron2..."
python3 -c "import detectron2; print(f'✅ Detectron2: {detectron2.__version__}')"

echo "🌐 Starting optimized FastAPI server on port 80..."
echo "📊 Full PDF layout analysis enabled with A10G GPU acceleration"

# Start the optimized server using the original app.py with detectron2 environment
sudo -E env PATH="$PATH" PYTHONPATH="$PYTHONPATH" /home/ubuntu/envs/d2/bin/python3 src/app.py --host 0.0.0.0 --port 80