#!/usr/bin/env python3
"""
A10G GPU-Optimized PDF Layout Analysis Server
Direct port of src/app.py with A10G optimizations
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

# Set up Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from catch_exceptions import catch_exceptions
    from configuration import service_logger, OCR_SOURCE
    from ocr.languages import supported_languages
    from ocr.ocr_pdf import ocr_pdf
    from pdf_layout_analysis.get_xml import get_xml
    from pdf_layout_analysis.run_pdf_layout_analysis import analyze_pdf
    from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
    from text_extraction.get_text_extraction import get_text_extraction
    from toc.get_toc import get_toc
    from visualization.get_visualization import get_visualization
    
    FULL_ANALYSIS_AVAILABLE = True
    service_logger.info(f"A10G GPU Analysis - Full analysis modules loaded successfully")
    
except ImportError as e:
    service_logger.error(f"Failed to import analysis modules: {e}")
    FULL_ANALYSIS_AVAILABLE = False
    
    # Simple fallback exception handler
    def catch_exceptions(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return JSONResponse(
                    content={"error": str(e), "status": "failed"},
                    status_code=500
                )
        return wrapper

service_logger.info(f"PyTorch GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    service_logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    service_logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

app = FastAPI(
    title="A10G GPU PDF Layout Analysis",
    description="Native A10G GPU-optimized PDF document layout analysis service",
    version="2.0.0-native"
)

@app.get("/")
async def root():
    gpu_info = f"A10G GPU: {torch.cuda.is_available()}"
    if torch.cuda.is_available():
        gpu_info += f" ({torch.cuda.get_device_name(0)})"
    status = "Full Analysis Available" if FULL_ANALYSIS_AVAILABLE else "Fallback Mode"
    return f"{sys.version} | {gpu_info} | {status}"

@app.get("/info")
async def info():
    gpu_stats = {}
    if torch.cuda.is_available():
        gpu_stats = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.3f} GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.3f} GB",
            "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
        }
    
    info_data = {
        "sys": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_stats": gpu_stats,
        "full_analysis_available": FULL_ANALYSIS_AVAILABLE,
        "optimization_status": "A10G Native Deployment (No Docker)",
        "performance_target": "3-4x faster than Docker baseline"
    }
    
    try:
        info_data["tesseract_version"] = subprocess.run(
            "tesseract --version", shell=True, text=True, capture_output=True
        ).stdout.split('\n')[0]
    except:
        info_data["tesseract_version"] = "Not available"
    
    try:
        info_data["supported_languages"] = supported_languages()
    except:
        info_data["supported_languages"] = []
    
    return info_data

@app.get("/error")
async def error():
    raise FileNotFoundError("This is a test error from the error endpoint")

if FULL_ANALYSIS_AVAILABLE:
    
    @app.post("/")
    @catch_exceptions
    async def run(file: UploadFile = File(...), fast: bool = Form(False), extraction_format: str = Form("")):
        """Main analysis endpoint - identical to original Docker service"""
        if fast:
            return await run_in_threadpool(analyze_pdf_fast, file.file.read(), "", extraction_format)
        return await run_in_threadpool(analyze_pdf, file.file.read(), "", extraction_format)

    @app.post("/save_xml/{xml_file_name}")
    @catch_exceptions
    async def analyze_and_save_xml(file: UploadFile = File(...), xml_file_name: str | None = None, fast: bool = Form(False)):
        xml_file_name = xml_file_name if xml_file_name.endswith(".xml") else f"{xml_file_name}.xml"
        if fast:
            return await run_in_threadpool(analyze_pdf_fast, file.file.read(), xml_file_name, "")
        return await run_in_threadpool(analyze_pdf, file.file.read(), xml_file_name, "")

    @app.get("/get_xml/{xml_file_name}", response_class=PlainTextResponse)
    @catch_exceptions
    async def get_xml_by_name(xml_file_name: str):
        xml_file_name = xml_file_name if xml_file_name.endswith(".xml") else f"{xml_file_name}.xml"
        return await run_in_threadpool(get_xml, xml_file_name)

    @app.post("/toc")
    @catch_exceptions
    async def get_toc_endpoint(file: UploadFile = File(...), fast: bool = Form(False)):
        return await run_in_threadpool(get_toc, file, fast)

    @app.post("/toc_legacy_uwazi_compatible")
    @catch_exceptions
    async def toc_legacy_uwazi_compatible(file: UploadFile = File(...)):
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
        return await run_in_threadpool(get_text_extraction, file, fast, types)

    @app.post("/visualize")
    @catch_exceptions
    async def get_visualization_endpoint(file: UploadFile = File(...), fast: bool = Form(False)):
        return await run_in_threadpool(get_visualization, file, fast)

    @app.post("/ocr")
    @catch_exceptions
    async def ocr_pdf_sync(file: UploadFile = File(...), language: str = Form("en")):
        namespace = "sync_pdfs"
        path = Path(OCR_SOURCE, namespace, file.filename)
        os.makedirs(path.parent, exist_ok=True)
        path.write_bytes(file.file.read())
        processed_pdf_filepath = ocr_pdf(file.filename, namespace, language)
        return FileResponse(path=processed_pdf_filepath, media_type="application/pdf")

else:
    
    @app.post("/")
    @catch_exceptions 
    async def fallback_analysis(file: UploadFile = File(...), fast: bool = Form(False), extraction_format: str = Form("")):
        """Fallback analysis when full modules not available"""
        return {
            "error": "Full analysis modules not available",
            "status": "Requires detectron2 and other dependencies",
            "filename": file.filename,
            "message": "Please install missing dependencies or use Docker image",
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }

if __name__ == "__main__":
    import uvicorn
    
    if torch.cuda.is_available():
        service_logger.info("🚀 A10G GPU detected")
        service_logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        service_logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        service_logger.info(f"Compute: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    else:
        service_logger.warning("⚠️  GPU not available")
    
    if FULL_ANALYSIS_AVAILABLE:
        service_logger.info("✅ Full PDF layout analysis available")
    else:
        service_logger.warning("⚠️  Limited functionality - missing dependencies")
    
    service_logger.info("🌐 Starting A10G Native PDF Analysis Server on port 80")
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="info")