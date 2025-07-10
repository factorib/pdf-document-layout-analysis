#!/usr/bin/env python3
"""
NVIDIA A10G Optimized PDF Layout Analysis Server
Provides full document layout analysis with GPU acceleration
"""

import os
import sys
import time
import subprocess
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from starlette.concurrency import run_in_threadpool

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from catch_exceptions import catch_exceptions
from configuration import service_logger, OCR_SOURCE
from ocr.languages import supported_languages
from ocr.ocr_pdf import ocr_pdf
from pdf_layout_analysis.get_xml import get_xml
from pdf_layout_analysis.run_pdf_layout_analysis_optimized import analyze_pdf_optimized
from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
from text_extraction.get_text_extraction import get_text_extraction
from toc.get_toc import get_toc
from visualization.get_visualization import get_visualization

service_logger.info(f"A10G GPU Optimization - PyTorch GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    service_logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    service_logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

app = FastAPI(
    title="A10G GPU-Optimized PDF Layout Analysis",
    description="Full document layout analysis with NVIDIA A10G GPU acceleration",
    version="2.0.0-a10g"
)

@app.get("/")
async def root():
    gpu_info = f"A10G GPU: {torch.cuda.is_available()}"
    if torch.cuda.is_available():
        gpu_info += f" ({torch.cuda.get_device_name(0)})"
    return sys.version + " " + gpu_info

@app.get("/info")
async def info():
    gpu_stats = {}
    if torch.cuda.is_available():
        gpu_stats = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.3f} GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.3f} GB"
        }
    
    return {
        "sys": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_stats": gpu_stats,
        "tesseract_version": subprocess.run("tesseract --version", shell=True, text=True, capture_output=True).stdout.split('\n')[0] if subprocess.run("tesseract --version", shell=True, text=True, capture_output=True).returncode == 0 else "Not available",
        "supported_languages": supported_languages() if 'supported_languages' in globals() else [],
        "optimization_features": [
            "A10G Tensor Core Acceleration",
            "Mixed Precision FP16",
            "GPU Memory Optimization",
            "Native Deployment (No Docker)",
            "CUDA 12.1 Support"
        ]
    }

@app.get("/gpu_benchmark")
async def gpu_benchmark():
    """A10G GPU benchmark test"""
    if not torch.cuda.is_available():
        return {"error": "GPU not available"}
    
    # Simple GPU benchmark
    start_time = time.time()
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    end_time = time.time()
    
    return {
        "gpu": torch.cuda.get_device_name(0),
        "benchmark_time": f"{end_time - start_time:.3f}s",
        "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB",
        "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
        "tensor_cores": "Active" if "A10" in torch.cuda.get_device_name(0) else "Not available"
    }

@app.post("/")
@catch_exceptions
async def analyze_pdf_endpoint(file: UploadFile = File(...), fast: bool = Form(False), extraction_format: str = Form("")):
    """Main PDF analysis endpoint - full compatibility with original service"""
    if fast:
        return await run_in_threadpool(analyze_pdf_fast, file.file.read(), "", extraction_format)
    
    # Use the A10G optimized analysis
    return await run_in_threadpool(analyze_pdf_optimized, file.file.read(), "", extraction_format)

@app.post("/save_xml/{xml_file_name}")
@catch_exceptions
async def analyze_and_save_xml(file: UploadFile = File(...), xml_file_name: str | None = None, fast: bool = Form(False)):
    """Save analysis results as XML"""
    xml_file_name = xml_file_name if xml_file_name.endswith(".xml") else f"{xml_file_name}.xml"
    if fast:
        return await run_in_threadpool(analyze_pdf_fast, file.file.read(), xml_file_name, "")
    return await run_in_threadpool(analyze_pdf_optimized, file.file.read(), xml_file_name, "")

@app.get("/get_xml/{xml_file_name}", response_class=PlainTextResponse)
@catch_exceptions
async def get_xml_by_name(xml_file_name: str):
    """Retrieve saved XML analysis"""
    xml_file_name = xml_file_name if xml_file_name.endswith(".xml") else f"{xml_file_name}.xml"
    return await run_in_threadpool(get_xml, xml_file_name)

@app.post("/toc")
@catch_exceptions
async def get_toc_endpoint(file: UploadFile = File(...), fast: bool = Form(False)):
    """Extract table of contents"""
    return await run_in_threadpool(get_toc, file, fast)

@app.post("/toc_legacy_uwazi_compatible")
@catch_exceptions
async def toc_legacy_uwazi_compatible(file: UploadFile = File(...)):
    """Legacy TOC format for Uwazi compatibility"""
    toc = await run_in_threadpool(get_toc, file, True)
    toc_compatible = []
    for toc_item in toc:
        toc_compatible.append(toc_item.copy())
        toc_compatible[-1]["bounding_box"]["left"] = int(toc_item["bounding_box"]["left"] / 0.75)
        toc_compatible[-1]["bounding_box"]["top"] = int(toc_item["bounding_box"]["top"] / 0.75)
        toc_compatible[-1]["bounding_box"]["width"] = int(toc_item["bounding_box"]["width"] / 0.75)
        toc_compatible[-1]["bounding_box"]["height"] = int(toc_item["bounding_box"]["height"] / 0.75)
        toc_compatible[-1]["selectionRectangles"] = [toc_compatible[-1]["bounding_box"]]
        del toc_compatible[-1]["bounding_box"]
    return toc_compatible

@app.post("/text")
@catch_exceptions
async def get_text_endpoint(file: UploadFile = File(...), fast: bool = Form(False), types: str = Form("all")):
    """Extract text from PDF"""
    return await run_in_threadpool(get_text_extraction, file, fast, types)

@app.post("/visualize")
@catch_exceptions
async def get_visualization_endpoint(file: UploadFile = File(...), fast: bool = Form(False)):
    """Get PDF visualization with detected elements"""
    return await run_in_threadpool(get_visualization, file, fast)

@app.post("/ocr")
@catch_exceptions
async def ocr_pdf_sync(file: UploadFile = File(...), language: str = Form("en")):
    """OCR processing"""
    namespace = "sync_pdfs"
    path = Path(OCR_SOURCE, namespace, file.filename)
    os.makedirs(path.parent, exist_ok=True)
    path.write_bytes(file.file.read())
    processed_pdf_filepath = ocr_pdf(file.filename, namespace, language)
    return FileResponse(path=processed_pdf_filepath, media_type="application/pdf")

@app.get("/error")
async def error():
    """Test error endpoint"""
    raise FileNotFoundError("This is a test error from the error endpoint")

if __name__ == "__main__":
    import uvicorn
    
    # Test GPU availability
    if torch.cuda.is_available():
        service_logger.info("🚀 A10G GPU detected and ready for optimization")
        service_logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        service_logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        service_logger.info("Mixed precision and Tensor Core acceleration enabled")
    else:
        service_logger.warning("⚠️  GPU not available - falling back to CPU")
    
    service_logger.info("🌐 Starting A10G-optimized PDF Layout Analysis Server")
    service_logger.info("Endpoints:")
    service_logger.info("  - GET  /                        - Status")
    service_logger.info("  - GET  /info                    - System info and GPU stats")
    service_logger.info("  - GET  /gpu_benchmark           - A10G benchmark test")
    service_logger.info("  - POST /                        - Full PDF analysis (optimized)")
    service_logger.info("  - POST /save_xml/{name}         - Save analysis as XML")
    service_logger.info("  - GET  /get_xml/{name}          - Retrieve saved XML")
    service_logger.info("  - POST /toc                     - Extract table of contents") 
    service_logger.info("  - POST /text                    - Extract text")
    service_logger.info("  - POST /visualize               - Get visualization")
    service_logger.info("  - POST /ocr                     - OCR processing")
    
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")