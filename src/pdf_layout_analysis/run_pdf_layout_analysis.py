import tempfile
import uuid
import time
from os.path import join
from pathlib import Path
from typing import AnyStr
from data_model.SegmentBox import SegmentBox
from ditod.VGTTrainer import VGTTrainer
from extraction_formats.extract_formula_formats import extract_formula_format
from extraction_formats.extract_table_formats import extract_table_format
from vgt.get_json_annotations import get_annotations
from vgt.get_model_configuration import get_model_configuration
from vgt.get_most_probable_pdf_segments import get_most_probable_pdf_segments
from vgt.get_reading_orders import get_reading_orders
from data_model.PdfImages import PdfImages
from src.configuration import service_logger, JSON_TEST_FILE_PATH, IMAGES_ROOT_PATH
from vgt.create_word_grid import create_word_grid, remove_word_grids
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

# Global variables for lazy loading
_configuration = None
_model = None

def get_or_load_model():
    """Lazy loading of model to avoid loading during import"""
    global _configuration, _model
    if _configuration is None:
        _configuration = get_model_configuration()
    if _model is None:
        _model = VGTTrainer.build_model(_configuration)
        DetectionCheckpointer(_model, save_dir=_configuration.OUTPUT_DIR).resume_or_load(_configuration.MODEL.WEIGHTS, resume=True)
    return _configuration, _model


def get_file_path(file_name, extension):
    return join(tempfile.gettempdir(), file_name + "." + extension)


def pdf_content_to_pdf_path(file_content, unique_id=None):
    if unique_id is None:
        file_id = str(uuid.uuid1())
    else:
        file_id = f"pdf_{unique_id}_{str(uuid.uuid1())[:8]}"

    pdf_path = Path(get_file_path(file_id, "pdf"))
    pdf_path.write_bytes(file_content)

    return pdf_path


def register_data():
    try:
        DatasetCatalog.remove("predict_data")
    except KeyError:
        pass

    register_coco_instances("predict_data", {}, JSON_TEST_FILE_PATH, IMAGES_ROOT_PATH)


def predict_doclaynet():
    configuration, model = get_or_load_model()
    register_data()
    VGTTrainer.test(configuration, model)


def analyze_pdf(file: AnyStr, xml_file_name: str, extraction_format: str = "", keep_pdf: bool = False) -> list[dict]:
    import uuid
    import threading
    
    # Create unique identifier for this request
    request_id = str(uuid.uuid4())[:8] 
    thread_id = threading.current_thread().ident
    unique_id = f"{request_id}_{thread_id}"
    
    # Start overall timing
    start_time = time.time()
    timing_data = {}
    
    try:
        # Stage 1: PDF Path Creation
        stage_start = time.time()
        pdf_path = pdf_content_to_pdf_path(file, unique_id)
        timing_data['pdf_path_creation'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] Starting PDF analysis")
        
        # Stage 2: PDF Images Processing
        stage_start = time.time()
        pdf_images_list: list[PdfImages] = [PdfImages.from_pdf_path(pdf_path, "", xml_file_name)]
        timing_data['pdf_images_creation'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] PDF images created in {timing_data['pdf_images_creation']:.3f}s")
        
        # Stage 3: Word Grid Creation
        stage_start = time.time()
        create_word_grid([pdf_images.pdf_features for pdf_images in pdf_images_list])
        timing_data['word_grid_creation'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] Word grid created in {timing_data['word_grid_creation']:.3f}s")
        
        # Stage 4: Annotation Generation
        stage_start = time.time()
        get_annotations(pdf_images_list)
        timing_data['annotation_generation'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] Annotations generated in {timing_data['annotation_generation']:.3f}s")
        
        # Stage 5: GPU Inference (Critical Section)
        stage_start = time.time()
        service_logger.info(f"[{unique_id}] Running GPU inference")
        predict_doclaynet()
        timing_data['gpu_inference'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] GPU inference completed in {timing_data['gpu_inference']:.3f}s")
        
        # Stage 6: Segment Extraction
        stage_start = time.time()
        predicted_segments = get_most_probable_pdf_segments("doclaynet", pdf_images_list, False)
        timing_data['segment_extraction'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] Segment extraction completed in {timing_data['segment_extraction']:.3f}s")
        
        # Stage 7: Reading Order Analysis
        stage_start = time.time()
        predicted_segments = get_reading_orders(pdf_images_list, predicted_segments)
        timing_data['reading_order_analysis'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] Reading order analysis completed in {timing_data['reading_order_analysis']:.3f}s")
        
        # Stage 8: Formula Format Extraction
        stage_start = time.time()
        extract_formula_format(pdf_images_list[0], predicted_segments)
        timing_data['formula_extraction'] = time.time() - stage_start
        service_logger.info(f"[{unique_id}] Formula extraction completed in {timing_data['formula_extraction']:.3f}s")
        
        # Stage 9: Table Format Extraction (if needed)
        if extraction_format:
            stage_start = time.time()
            extract_table_format(pdf_images_list[0], predicted_segments, extraction_format)
            timing_data['table_extraction'] = time.time() - stage_start
            service_logger.info(f"[{unique_id}] Table extraction completed in {timing_data['table_extraction']:.3f}s")
        
        # Stage 10: Result Conversion
        stage_start = time.time()
        result = [
            SegmentBox.from_pdf_segment(pdf_segment, pdf_images_list[0].pdf_features.pages).to_dict()
            for pdf_segment in predicted_segments
        ]
        timing_data['result_conversion'] = time.time() - stage_start
        
        # Calculate total time
        total_time = time.time() - start_time
        timing_data['total_time'] = total_time
        
        # Log comprehensive timing summary
        service_logger.info(f"[{unique_id}] TIMING SUMMARY - Total: {total_time:.3f}s")
        service_logger.info(f"[{unique_id}] ├── PDF Path Creation: {timing_data['pdf_path_creation']:.3f}s ({timing_data['pdf_path_creation']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] ├── PDF Images Creation: {timing_data['pdf_images_creation']:.3f}s ({timing_data['pdf_images_creation']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] ├── Word Grid Creation: {timing_data['word_grid_creation']:.3f}s ({timing_data['word_grid_creation']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] ├── Annotation Generation: {timing_data['annotation_generation']:.3f}s ({timing_data['annotation_generation']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] ├── GPU Inference: {timing_data['gpu_inference']:.3f}s ({timing_data['gpu_inference']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] ├── Segment Extraction: {timing_data['segment_extraction']:.3f}s ({timing_data['segment_extraction']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] ├── Reading Order Analysis: {timing_data['reading_order_analysis']:.3f}s ({timing_data['reading_order_analysis']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] ├── Formula Extraction: {timing_data['formula_extraction']:.3f}s ({timing_data['formula_extraction']/total_time*100:.1f}%)")
        if extraction_format:
            service_logger.info(f"[{unique_id}] ├── Table Extraction: {timing_data['table_extraction']:.3f}s ({timing_data['table_extraction']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] └── Result Conversion: {timing_data['result_conversion']:.3f}s ({timing_data['result_conversion']/total_time*100:.1f}%)")
        service_logger.info(f"[{unique_id}] Analysis complete: {len(predicted_segments)} segments found")
        
        return result
        
    except Exception as e:
        service_logger.error(f"[{unique_id}] Analysis failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        try:
            remove_files()
            if not keep_pdf and 'pdf_path' in locals():
                pdf_path.unlink(missing_ok=True)
        except Exception as cleanup_error:
            service_logger.warning(f"[{unique_id}] Cleanup warning: {cleanup_error}")


def remove_files():
    PdfImages.remove_images()
    remove_word_grids()
