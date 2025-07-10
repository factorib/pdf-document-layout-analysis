from pathlib import Path
from tempfile import gettempdir
from fastapi import UploadFile
from starlette.responses import FileResponse
from pdf_layout_analysis.run_pdf_layout_analysis import analyze_pdf
from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
from visualization.save_output_to_pdf import save_output_to_pdf
from glob import glob
from os.path import getctime, join
import time
import logging

service_logger = logging.getLogger(__name__)


def get_visualization(file: UploadFile, fast: bool):
    # Start visualization timing
    start_time = time.time()
    timing_data = {}
    
    # Generate unique ID for this visualization request
    import uuid
    viz_id = str(uuid.uuid4())[:8]
    
    service_logger.info(f"[VIZ-{viz_id}] Starting visualization generation (fast={fast})")
    
    # Stage 1: File Content Reading
    stage_start = time.time()
    file_content = file.file.read()
    timing_data['file_read'] = time.time() - stage_start
    service_logger.info(f"[VIZ-{viz_id}] File content read in {timing_data['file_read']:.3f}s")
    
    # Stage 2: PDF Analysis
    stage_start = time.time()
    if fast:
        segment_boxes = analyze_pdf_fast(file_content, "", "", True)
    else:
        segment_boxes = analyze_pdf(file_content, "", "", True)
    timing_data['pdf_analysis'] = time.time() - stage_start
    service_logger.info(f"[VIZ-{viz_id}] PDF analysis completed in {timing_data['pdf_analysis']:.3f}s")
    
    # Stage 3: PDF Path Discovery
    stage_start = time.time()
    pdf_path = max(glob(join(gettempdir(), "*.pdf")), key=getctime)
    timing_data['pdf_path_discovery'] = time.time() - stage_start
    service_logger.info(f"[VIZ-{viz_id}] PDF path discovered in {timing_data['pdf_path_discovery']:.3f}s")
    
    # Stage 4: PDF Annotation Generation
    stage_start = time.time()
    save_output_to_pdf(pdf_path, segment_boxes)
    timing_data['pdf_annotation'] = time.time() - stage_start
    service_logger.info(f"[VIZ-{viz_id}] PDF annotation completed in {timing_data['pdf_annotation']:.3f}s")
    
    # Stage 5: File Response Creation
    stage_start = time.time()
    file_response = FileResponse(pdf_path, media_type="application/pdf", filename=Path(pdf_path).name)
    timing_data['response_creation'] = time.time() - stage_start
    
    # Calculate total time
    total_time = time.time() - start_time
    timing_data['total_time'] = total_time
    
    # Log comprehensive timing summary
    service_logger.info(f"[VIZ-{viz_id}] VISUALIZATION TIMING SUMMARY - Total: {total_time:.3f}s")
    service_logger.info(f"[VIZ-{viz_id}] ├── File Read: {timing_data['file_read']:.3f}s ({timing_data['file_read']/total_time*100:.1f}%)")
    service_logger.info(f"[VIZ-{viz_id}] ├── PDF Analysis: {timing_data['pdf_analysis']:.3f}s ({timing_data['pdf_analysis']/total_time*100:.1f}%)")
    service_logger.info(f"[VIZ-{viz_id}] ├── PDF Path Discovery: {timing_data['pdf_path_discovery']:.3f}s ({timing_data['pdf_path_discovery']/total_time*100:.1f}%)")
    service_logger.info(f"[VIZ-{viz_id}] ├── PDF Annotation: {timing_data['pdf_annotation']:.3f}s ({timing_data['pdf_annotation']/total_time*100:.1f}%)")
    service_logger.info(f"[VIZ-{viz_id}] └── Response Creation: {timing_data['response_creation']:.3f}s ({timing_data['response_creation']/total_time*100:.1f}%)")
    service_logger.info(f"[VIZ-{viz_id}] Visualization complete: {len(segment_boxes)} segments annotated")
    
    return file_response
