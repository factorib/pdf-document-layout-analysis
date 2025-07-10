import time
from typing import Optional

import torch
from PIL import Image
from struct_eqtable import build_model

from configuration import service_logger
from data_model.PdfImages import PdfImages
from fast_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TokenType import TokenType

# Global cached model instance to avoid repeated initialization
_table_model = None


def get_table_format(
    model,
    raw_image: Image = None,
    image_path: str = "",
    max_waiting_time: int = 1000,
    extraction_format: str = "latex",
) -> str:
    from pypandoc import convert_text

    if not raw_image:
        raw_image = Image.open(image_path)

    start_time = time.time()
    with torch.no_grad():
        output = model(raw_image)

    cost_time = time.time() - start_time

    if cost_time >= max_waiting_time:
        warn_log = (
            f"The table extraction model inference time exceeds the maximum waiting time {max_waiting_time} seconds.\n"
            "Please increase the maximum waiting time or model may not support the type of input table image"
        )
        service_logger.info(warn_log)

    for i, latex_code in enumerate(output):
        for tgt_fmt in [extraction_format]:
            tgt_code = convert_text(latex_code, tgt_fmt, format="latex") if tgt_fmt != "latex" else latex_code
            return tgt_code


def get_cached_table_model():
    """Get cached table model instance to avoid repeated initialization."""
    global _table_model
    if _table_model is None:
        _table_model = _build_table_model()
    return _table_model


def _build_table_model():
    """Build table model instance (internal function)."""
    ckpt_path: str = "U4R/StructTable-base"
    max_new_tokens: int = 2048
    max_waiting_time: int = 1000
    use_cpu: bool = False
    tensorrt_path: Optional[str] = None
    model = build_model(ckpt_path, max_new_tokens=max_new_tokens, max_time=max_waiting_time, tensorrt_path=tensorrt_path)
    if not use_cpu and tensorrt_path is None:
        try:
            model = model.cuda()
        except RuntimeError:
            pass
    return model


def get_model():
    """Legacy function - use get_cached_table_model() instead."""
    return get_cached_table_model()


def extract_table_format(pdf_images: PdfImages, predicted_segments: list[PdfSegment], extraction_format: str):
    start_time = time.time()
    
    table_segments = [
        (index, segment) for index, segment in enumerate(predicted_segments) if segment.segment_type == TokenType.TABLE
    ]
    if not table_segments:
        service_logger.info(f"Table extraction: No table segments found - completed in {time.time() - start_time:.3f}s")
        return

    service_logger.info(f"Table extraction: Found {len(table_segments)} table segments to process")

    # Model initialization timing
    model_start = time.time()
    model = get_cached_table_model()
    model_time = time.time() - model_start
    service_logger.info(f"Table extraction: Model initialization completed in {model_time:.3f}s")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for index, table_segment in table_segments:
        segment_start = time.time()
        
        # Skip segments with invalid dimensions
        if table_segment.bounding_box.width <= 0 or table_segment.bounding_box.height <= 0:
            skipped_count += 1
            continue
            
        # Image processing timing
        image_start = time.time()
        page_image: Image = pdf_images.pdf_images[table_segment.page_number - 1]
        left, top = table_segment.bounding_box.left, table_segment.bounding_box.top
        width, height = table_segment.bounding_box.width, table_segment.bounding_box.height
        table_image = page_image.crop((left, top, left + width, top + height))
        image_time = time.time() - image_start
        
        # Table inference timing
        inference_start = time.time()
        try:
            extracted_table = get_table_format(model, raw_image=table_image, extraction_format=extraction_format)
            inference_time = time.time() - inference_start
            predicted_segments[index].text_content = extracted_table
            processed_count += 1
            
            segment_time = time.time() - segment_start
            service_logger.info(f"Table extraction: Segment {index} processed in {segment_time:.3f}s (image:{image_time:.3f}s, inference:{inference_time:.3f}s)")
            
        except RuntimeError as e:
            error_count += 1
            inference_time = time.time() - inference_start
            segment_time = time.time() - segment_start
            service_logger.warning(f"Table extraction: Segment {index} failed in {segment_time:.3f}s (image:{image_time:.3f}s, inference:{inference_time:.3f}s) - {e}")
            continue

    total_time = time.time() - start_time
    service_logger.info(f"Table extraction: SUMMARY - Total:{total_time:.3f}s, Processed:{processed_count}, Skipped:{skipped_count}, Errors:{error_count}")
