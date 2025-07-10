#!/usr/bin/env python3
"""
A10G GPU-Optimized PDF Layout Analysis Server
Uses detectron2 environment with full PDF analysis capabilities
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables for detectron2 compatibility
os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")

import uvicorn
from src.app import app

if __name__ == "__main__":
    print("🚀 Starting A10G GPU-Optimized PDF Layout Analysis Server")
    print("✅ Detectron2 environment active")
    print("✅ Full PDF analysis enabled")
    print("🌐 Server starting on port 80...")
    
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")