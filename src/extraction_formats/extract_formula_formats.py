import io
import time
from PIL.Image import Image
from rapid_latex_ocr import LaTeXOCR
from data_model.PdfImages import PdfImages
from fast_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TokenType import TokenType
from configuration import service_logger

# Global cached model instance to avoid repeated initialization
_latex_model = None


def has_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" for char in text)


def get_latex_format(model: LaTeXOCR, formula_image: Image):
    buffer = io.BytesIO()
    formula_image.save(buffer, format="jpeg")
    image_bytes = buffer.getvalue()
    result, elapsed_time = model(image_bytes)
    return result


def get_cached_latex_model():
    """Get cached LaTeX OCR model instance to avoid repeated initialization."""
    global _latex_model
    if _latex_model is None:
        _latex_model = LaTeXOCR()
    return _latex_model


def extract_formula_format(pdf_images: PdfImages, predicted_segments: list[PdfSegment]):
    start_time = time.time()
    
    formula_segments = [
        (index, segment) for index, segment in enumerate(predicted_segments) if segment.segment_type == TokenType.FORMULA
    ]
    if not formula_segments:
        service_logger.info(f"Formula extraction: No formula segments found - completed in {time.time() - start_time:.3f}s")
        return

    service_logger.info(f"Formula extraction: Found {len(formula_segments)} formula segments to process")

    # Model initialization timing
    model_start = time.time()
    model = get_cached_latex_model()
    model_time = time.time() - model_start
    service_logger.info(f"Formula extraction: Model initialization completed in {model_time:.3f}s")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for index, formula_segment in formula_segments:
        segment_start = time.time()
        
        # Skip empty or non-existent text content
        if not formula_segment.text_content or not formula_segment.text_content.strip():
            skipped_count += 1
            continue
        if has_arabic(formula_segment.text_content):
            skipped_count += 1
            continue
            
        # Image processing timing
        image_start = time.time()
        page_image: Image = pdf_images.pdf_images[formula_segment.page_number - 1]
        left, top = formula_segment.bounding_box.left, formula_segment.bounding_box.top
        width, height = formula_segment.bounding_box.width, formula_segment.bounding_box.height
        formula_image = page_image.crop((left, top, left + width, top + height))
        image_time = time.time() - image_start
        
        # OCR inference timing
        ocr_start = time.time()
        try:
            extracted_formula = get_latex_format(model, formula_image)
            ocr_time = time.time() - ocr_start
            predicted_segments[index].text_content = extracted_formula
            processed_count += 1
            
            segment_time = time.time() - segment_start
            service_logger.info(f"Formula extraction: Segment {index} processed in {segment_time:.3f}s (image:{image_time:.3f}s, ocr:{ocr_time:.3f}s)")
            
        except (ValueError, RuntimeError) as e:
            error_count += 1
            ocr_time = time.time() - ocr_start
            segment_time = time.time() - segment_start
            service_logger.warning(f"Formula extraction: Segment {index} failed in {segment_time:.3f}s (image:{image_time:.3f}s, ocr:{ocr_time:.3f}s) - {e}")
            continue

    total_time = time.time() - start_time
    service_logger.info(f"Formula extraction: SUMMARY - Total:{total_time:.3f}s, Processed:{processed_count}, Skipped:{skipped_count}, Errors:{error_count}")
