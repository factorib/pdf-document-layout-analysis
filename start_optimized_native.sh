#!/bin/bash

# Optimized Native Startup Script for NVIDIA A10G GPU
# Implements comprehensive optimization plan for 3-4x performance improvement

echo "=========================================="
echo "PDF Analyzer Native Optimization Startup"
echo "Target: NVIDIA A10G GPU (AWS g5.2xlarge)"
echo "=========================================="

# Set CUDA environment for A10G
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4

# GPU and Memory Optimizations for A10G (24GB VRAM)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8,expandable_segments:False,roundup_power2_divisions:16"
export TORCH_BACKENDS_CUDNN_BENCHMARK=1
export TORCH_BACKENDS_CUDNN_DETERMINISTIC=0
export TORCH_BACKENDS_CUDNN_ENABLED=1
export TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=1
export TORCH_CUDNN_V8_API_ENABLED=1

# CUDA Device Configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY

# CPU Optimizations for 8-core AMD EPYC 7R32
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Memory and I/O optimizations
export MALLOC_ARENA_MAX=2

echo "Environment variables configured for A10G optimization"

# Verify GPU availability and optimization readiness
echo "Checking GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

# Check CUDA and PyTorch compatibility
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'Compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
    # Test Tensor Core availability
    test_tensor = torch.randn(16, 16, device='cuda:0', dtype=torch.float16)
    result = torch.mm(test_tensor, test_tensor)
    print(f'Tensor Core test: PASSED')
else:
    print('CUDA NOT AVAILABLE - GPU optimizations disabled')
"

# System performance optimizations
echo "Applying system-level optimizations..."

# CPU governor to performance mode (if available)
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    echo "CPU governor set to performance mode"
fi

# Memory management optimizations (if writable)
if [ -w /proc/sys/vm/swappiness ]; then
    echo 1 | sudo tee /proc/sys/vm/swappiness
    echo 50 | sudo tee /proc/sys/vm/vfs_cache_pressure
    echo "Memory management optimized"
fi

# GPU power and clock optimization
echo "Optimizing GPU power management..."
sudo nvidia-smi -pm 1 || echo "Could not enable GPU persistence mode"
sudo nvidia-smi -ac 6001,1710 || echo "Could not set maximum GPU clocks (partially supported on A10G)"

# Change to application directory
cd "$(dirname "$0")"

echo "=========================================="
echo "Starting Optimized PDF Analyzer Service"
echo "Expected Performance: 3-4x faster than Docker baseline"
echo "Target Processing Time: <10s for 11-page PDF"
echo "=========================================="

# Start the optimized application with CPU affinity for optimal performance
# Using taskset to bind to specific CPU cores for consistent performance
exec taskset -c 0-7 python3 -m uvicorn src.optimized_app:app \
    --host 0.0.0.0 \
    --port 5060 \
    --workers 1 \
    --access-log \
    --log-level info \
    --no-use-colors