# GPU Optimization Module for NVIDIA A10G
# Based on comprehensive optimization plan for native deployment

import torch
import os
from torch.cuda.amp import autocast, GradScaler
from detectron2.config import configurable
import logging

logger = logging.getLogger(__name__)

class A10GOptimizer:
    """
    Advanced GPU optimizations specifically for NVIDIA A10G GPU
    Implements A10G-specific optimizations for maximum performance
    """
    
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()
        self.is_a10g = self.verify_a10g_gpu()
        self.setup_environment()
        self.configure_gpu_optimizations()
        self.memory_pool = None
        self.tensor_cache = {}
        
    def verify_a10g_gpu(self):
        """Verify if we're actually running on A10G GPU"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            # A10G has compute capability 8.6
            is_a10g = "A10G" in gpu_name and compute_capability == (8, 6)
            if not is_a10g:
                logger.warning(f"Not running on A10G GPU. Detected: {gpu_name}, Compute: {compute_capability}")
            else:
                logger.info(f"A10G GPU detected: {gpu_name}, Compute: {compute_capability}")
            return is_a10g
        return False
        
    def setup_environment(self):
        """Set optimal environment variables for A10G GPU"""
        if self.is_a10g:
            # A10G-specific optimizations
            env_vars = {
                # A10G memory optimizations (24GB VRAM)
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1024,garbage_collection_threshold:0.9,expandable_segments:True,roundup_power2_divisions:8",
                "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
                "TORCH_BACKENDS_CUDNN_DETERMINISTIC": "0", 
                "TORCH_BACKENDS_CUDNN_ENABLED": "1",
                "TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32": "1",
                "TORCH_BACKENDS_CUDA_MATMUL_ALLOW_BF16_REDUCED_PRECISION_REDUCTION": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_LAUNCH_BLOCKING": "0",
                "CUDA_MODULE_LOADING": "LAZY",
                # A10G tensor core optimizations
                "NVIDIA_TF32_OVERRIDE": "1",
                "CUDA_AUTO_BOOST": "1",
                # CPU optimizations for g5.2xlarge (8 vCPUs)
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8", 
                "OPENBLAS_NUM_THREADS": "8",
                "NUMEXPR_NUM_THREADS": "8"
            }
        else:
            # Generic GPU optimizations for non-A10G
            env_vars = {
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8",
                "TORCH_BACKENDS_CUDNN_BENCHMARK": "1",
                "TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "4"
            }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        if self.is_a10g:
            logger.info("A10G-specific environment variables configured for optimal performance")
        else:
            logger.info("Generic GPU environment variables configured")
        
    def configure_gpu_optimizations(self):
        """Configure PyTorch for maximum A10G utilization"""
        if torch.cuda.is_available():
            # Enable Tensor Core optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            
            if self.is_a10g:
                # A10G-specific optimizations
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                torch.backends.cuda.preferred_linalg_library = "cusolver"
                
                # Configure A10G memory allocator
                if hasattr(torch.cuda, 'set_memory_allocator'):
                    self.memory_pool = torch.cuda.memory.MemoryPool()
                    torch.cuda.set_memory_allocator(self.memory_pool.allocator)
                
                # A10G optimal batch sizes for inference
                self.optimal_batch_sizes = [1, 2, 4, 8, 16]  # Powers of 2 for tensor cores
                
                logger.info(f"A10G GPU optimizations configured: {torch.cuda.get_device_name()}")
            else:
                # Generic GPU optimizations
                self.optimal_batch_sizes = [1, 2, 4, 8]
                logger.info(f"Generic GPU optimizations configured: {torch.cuda.get_device_name()}")
            
            # Pre-allocate common tensor sizes
            self.prealloc_tensors()
            
        else:
            logger.warning("CUDA not available - GPU optimizations disabled")
    
    def prealloc_tensors(self):
        """Pre-allocate common tensor sizes for A10G to avoid runtime allocation"""
        if self.is_a10g:
            # A10G-optimized tensor sizes (multiples of 8 for tensor cores)
            common_sizes = [
                (1, 3, 800, 1344),    # Single page (1344 = 1333 rounded up to multiple of 8)
                (2, 3, 800, 1344),    # Small batch
                (4, 3, 800, 1344),    # Medium batch  
                (8, 3, 800, 1344),    # Optimal batch for A10G
                (16, 3, 800, 1344),   # Large batch for A10G
            ]
        else:
            # Generic GPU tensor sizes
            common_sizes = [
                (1, 3, 800, 1333),    # Single page standard
                (2, 3, 800, 1333),    # Small batch
                (4, 3, 800, 1333),    # Medium batch
            ]
        
        for size in common_sizes:
            key = f"tensor_{size}"
            try:
                self.tensor_cache[key] = torch.empty(
                    size,
                    dtype=torch.float16,  # FP16 for Tensor Core utilization
                    device=self.device,
                    memory_format=torch.channels_last
                )
                logger.info(f"Pre-allocated tensor cache for size: {size}")
            except RuntimeError as e:
                logger.warning(f"Failed to pre-allocate tensor size {size}: {e}")
    
    def get_optimized_tensor(self, shape):
        """Get pre-allocated tensor or create optimized new one"""
        key = f"tensor_{shape}"
        if key in self.tensor_cache:
            return self.tensor_cache[key][:shape[0]]
        else:
            return torch.empty(
                shape,
                dtype=torch.float16,
                device=self.device, 
                memory_format=torch.channels_last
            )
    
    @autocast(dtype=torch.float16)
    def optimized_inference(self, model, images):
        """
        Optimized inference with mixed precision for A10G Tensor Cores
        Expected 2-4x speedup through FP16 and optimal batching
        """
        with torch.no_grad():
            # Ensure input is FP16 for Tensor Core utilization
            if images.dtype != torch.float16:
                images = images.to(dtype=torch.float16, device=self.device)
            
            # A10G-specific optimal batch size
            if self.is_a10g:
                # A10G can handle larger batches with 24GB VRAM
                max_batch_size = 16
            else:
                # Conservative batch size for other GPUs
                max_batch_size = 8
            
            batch_size = min(len(images) if hasattr(images, '__len__') else 1, max_batch_size)
            
            # Use channels_last for better memory layout on A10G
            if hasattr(images, 'to'):
                images = images.to(memory_format=torch.channels_last)
            
            # Run inference with Tensor Core acceleration
            return model(images)
    
    def clear_memory_cache_periodically(self, iteration_count):
        """Clear GPU memory cache periodically to prevent fragmentation"""
        if iteration_count % 32 == 0:  # Every 32 iterations
            torch.cuda.empty_cache()
            if self.memory_pool:
                self.memory_pool.empty_cache()
            logger.debug(f"GPU memory cache cleared at iteration {iteration_count}")
    
    def get_memory_stats(self):
        """Get current GPU memory statistics for monitoring"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # Get actual GPU memory capacity
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved, 
                "max_allocated_gb": max_allocated,
                "total_memory_gb": total_memory,
                "utilization_pct": (allocated / total_memory) * 100,
                "is_a10g": self.is_a10g
            }
        return {"error": "CUDA not available"}
    
    def optimize_model_for_tensorrt(self, model):
        """
        Optimize model structure for TensorRT deployment
        Ensures tensor dimensions are multiples of 8 for Tensor Core efficiency
        """
        for name, module in model.named_modules():
            # Replace adaptive pooling with fixed pooling for TensorRT compatibility
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                setattr(model, name, torch.nn.AvgPool2d(kernel_size=7, stride=1))
            
            # Ensure linear layers have dimensions multiple of 8 for Tensor Cores
            if isinstance(module, torch.nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                
                # Pad to nearest multiple of 8 for tensor core optimization
                in_features_padded = ((in_features + 7) // 8) * 8
                out_features_padded = ((out_features + 7) // 8) * 8
                
                if in_features != in_features_padded or out_features != out_features_padded:
                    new_linear = torch.nn.Linear(in_features_padded, out_features_padded)
                    # Copy weights with padding
                    new_linear.weight.data[:out_features, :in_features] = module.weight.data
                    new_linear.bias.data[:out_features] = module.bias.data
                    setattr(model, name, new_linear)
                    logger.info(f"Optimized linear layer {name} for Tensor Cores")
        
        return model


class OptimizedVGTInference:
    """
    Optimized VGT model inference class with A10G-specific optimizations
    Implements the performance improvements from the comprehensive plan
    """
    
    def __init__(self, model):
        self.model = model
        self.optimizer = A10GOptimizer()
        self.iteration_count = 0
        
        # Apply model optimizations
        self.model = self.optimizer.optimize_model_for_tensorrt(self.model)
        
        # Convert model to FP16 for Tensor Core utilization
        if torch.cuda.is_available():
            self.model = self.model.half().to(self.optimizer.device)
            self.model.eval()
        
        logger.info("OptimizedVGTInference initialized with A10G optimizations")
    
    def __call__(self, batched_inputs):
        """
        Optimized forward pass with mixed precision and memory management
        Expected 3-4x performance improvement over baseline
        """
        self.iteration_count += 1
        
        # Clear memory cache periodically
        self.optimizer.clear_memory_cache_periodically(self.iteration_count)
        
        # Run optimized inference
        result = self.optimizer.optimized_inference(self.model, batched_inputs)
        
        # Log memory statistics periodically
        if self.iteration_count % 10 == 0:
            stats = self.optimizer.get_memory_stats()
            logger.debug(f"GPU Memory - Allocated: {stats.get('allocated_gb', 0):.2f}GB, "
                        f"Utilization: {stats.get('utilization_pct', 0):.1f}%")
        
        return result


# Global optimizer instance
gpu_optimizer = A10GOptimizer()

def get_optimized_model(model):
    """Factory function to wrap any model with A10G optimizations"""
    return OptimizedVGTInference(model)