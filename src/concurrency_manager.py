#!/usr/bin/env python3
"""
Concurrency Manager for A10G GPU-Optimized PDF Analysis
Handles request queuing and resource isolation to prevent conflicts
"""

import asyncio
import threading
import time
from typing import Dict, Any, Callable
from functools import wraps
import uuid
import tempfile
import os
from pathlib import Path

class RequestManager:
    """Manages concurrent requests with proper resource isolation"""
    
    def __init__(self, max_concurrent: int = 1):
        # Use semaphore to limit concurrent GPU inference
        # GPU models are not thread-safe, so serialize inference
        self._gpu_semaphore = asyncio.Semaphore(max_concurrent)
        self._request_lock = threading.Lock()
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        
    async def process_request(self, request_func: Callable, *args, **kwargs):
        """Process a request with proper concurrency control"""
        request_id = str(uuid.uuid4())[:8]
        
        # Log request start
        with self._request_lock:
            self._active_requests[request_id] = {
                'start_time': time.time(),
                'status': 'waiting'
            }
        
        try:
            # Acquire GPU semaphore (serialize GPU operations)
            async with self._gpu_semaphore:
                with self._request_lock:
                    self._active_requests[request_id]['status'] = 'processing'
                    self._active_requests[request_id]['gpu_acquired'] = time.time()
                
                # Execute the actual request
                result = await asyncio.get_event_loop().run_in_executor(
                    None, request_func, *args, **kwargs
                )
                
                return result
                
        finally:
            # Clean up request tracking
            with self._request_lock:
                if request_id in self._active_requests:
                    del self._active_requests[request_id]

    def get_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        with self._request_lock:
            return {
                'active_requests': len(self._active_requests),
                'requests': dict(self._active_requests)
            }

# Global request manager instance
request_manager = RequestManager(max_concurrent=1)  # Serialize GPU requests

def gpu_synchronized(func):
    """Decorator to synchronize GPU operations"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await request_manager.process_request(func, *args, **kwargs)
    return wrapper

class IsolatedWorkspace:
    """Creates isolated workspace for each request"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.temp_dir = None
        self.original_paths = {}
        
    def __enter__(self):
        # Create isolated temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"pdf_analysis_{self.request_id}_"))
        
        # Create isolated subdirectories
        self.images_dir = self.temp_dir / "images"
        self.jsons_dir = self.temp_dir / "jsons"
        self.output_dir = self.temp_dir / "output"
        
        self.images_dir.mkdir(exist_ok=True)
        self.jsons_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store original global paths
        import src.configuration as config
        self.original_paths = {
            'IMAGES_ROOT_PATH': getattr(config, 'IMAGES_ROOT_PATH', None),
            'JSON_TEST_FILE_PATH': getattr(config, 'JSON_TEST_FILE_PATH', None),
        }
        
        # Override global paths for this request
        config.IMAGES_ROOT_PATH = str(self.images_dir)
        if hasattr(config, 'JSON_TEST_FILE_PATH'):
            config.JSON_TEST_FILE_PATH = str(self.jsons_dir / "test.json")
        
        # Create isolated COCO test file
        coco_data = {
            "images": [],
            "annotations": [], 
            "categories": [],
            "info": {
                "description": f"PDF Layout Analysis - Request {self.request_id}",
                "version": "1.0"
            }
        }
        
        import json
        with open(self.jsons_dir / "test.json", 'w') as f:
            json.dump(coco_data, f)
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original global paths
        import src.configuration as config
        for path_name, original_value in self.original_paths.items():
            if original_value is not None:
                setattr(config, path_name, original_value)
        
        # Clean up temporary directory
        import shutil
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass  # Ignore cleanup errors

def with_isolated_workspace(func):
    """Decorator to run function with isolated workspace"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())[:8]
        with IsolatedWorkspace(request_id):
            return func(*args, **kwargs)
    return wrapper