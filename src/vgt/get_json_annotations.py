import json
from os import makedirs
from pdf_features.PdfToken import PdfToken
from data_model.PdfImages import PdfImages
from configuration import DOCLAYNET_TYPE_BY_ID
from configuration import JSONS_ROOT_PATH, JSON_TEST_FILE_PATH
import time
import logging

service_logger = logging.getLogger(__name__)


def save_annotations_json(annotations: list, width_height: list, images: list):
    images_dict = [
        {
            "id": i,
            "file_name": image_id + ".jpg",
            "width": width_height[images.index(image_id)][0],
            "height": width_height[images.index(image_id)][1],
        }
        for i, image_id in enumerate(images)
    ]

    categories_dict = [{"id": key, "name": value} for key, value in DOCLAYNET_TYPE_BY_ID.items()]

    # Add the required "info" field for COCO format compatibility
    info_dict = {
        "description": "PDF Document Layout Analysis",
        "version": "1.0",
        "year": 2024,
        "contributor": "PDF Layout Analysis System",
        "date_created": "2024-01-01"
    }

    coco_dict = {"info": info_dict, "images": images_dict, "categories": categories_dict, "annotations": annotations}

    JSON_TEST_FILE_PATH.write_text(json.dumps(coco_dict))


def get_annotation(index: int, image_id: str, token: PdfToken):
    return {
        "area": 1,
        "iscrowd": 0,
        "score": 1,
        "image_id": image_id,
        "bbox": [token.bounding_box.left, token.bounding_box.top, token.bounding_box.width, token.bounding_box.height],
        "category_id": token.token_type.get_index(),
        "id": index,
    }


def get_annotations_for_document(annotations, images, index, pdf_images, width_height):
    for page_index, page in enumerate(pdf_images.pdf_features.pages):
        image_id = f"{pdf_images.pdf_features.file_name}_{page.page_number - 1}"
        images.append(image_id)
        width_height.append((pdf_images.pdf_images[page_index].width, pdf_images.pdf_images[page_index].height))

        for token in page.tokens:
            annotations.append(get_annotation(index, image_id, token))
            index += 1


def get_annotations(pdf_images_list: list[PdfImages]):
    # Start annotation timing
    start_time = time.time()
    timing_data = {}
    
    # Generate unique ID for this annotation process
    import uuid
    ann_id = str(uuid.uuid4())[:8]
    
    service_logger.info(f"[JSON-{ann_id}] Starting JSON annotation generation for {len(pdf_images_list)} documents")
    
    # Stage 1: Directory Setup
    stage_start = time.time()
    makedirs(JSONS_ROOT_PATH, exist_ok=True)
    timing_data['directory_setup'] = time.time() - stage_start
    service_logger.info(f"[JSON-{ann_id}] Directory setup completed in {timing_data['directory_setup']:.3f}s")
    
    # Stage 2: Data Structure Initialization
    stage_start = time.time()
    annotations = list()
    images = list()
    width_height = list()
    index = 0
    timing_data['data_initialization'] = time.time() - stage_start
    service_logger.info(f"[JSON-{ann_id}] Data initialization completed in {timing_data['data_initialization']:.3f}s")
    
    # Stage 3: Document Processing
    stage_start = time.time()
    total_tokens = 0
    for pdf_images in pdf_images_list:
        doc_start = time.time()
        get_annotations_for_document(annotations, images, index, pdf_images, width_height)
        doc_tokens = sum([len(page.tokens) for page in pdf_images.pdf_features.pages])
        total_tokens += doc_tokens
        index += doc_tokens
        doc_time = time.time() - doc_start
        service_logger.info(f"[JSON-{ann_id}] Document processed in {doc_time:.3f}s ({doc_tokens} tokens)")
    
    timing_data['document_processing'] = time.time() - stage_start
    service_logger.info(f"[JSON-{ann_id}] Document processing completed in {timing_data['document_processing']:.3f}s ({total_tokens} total tokens)")
    
    # Stage 4: JSON Saving
    stage_start = time.time()
    save_annotations_json(annotations, width_height, images)
    timing_data['json_saving'] = time.time() - stage_start
    service_logger.info(f"[JSON-{ann_id}] JSON saving completed in {timing_data['json_saving']:.3f}s")
    
    # Calculate total time
    total_time = time.time() - start_time
    timing_data['total_time'] = total_time
    
    # Log comprehensive timing summary
    service_logger.info(f"[JSON-{ann_id}] JSON ANNOTATION TIMING SUMMARY - Total: {total_time:.3f}s")
    service_logger.info(f"[JSON-{ann_id}] ├── Directory Setup: {timing_data['directory_setup']:.3f}s ({timing_data['directory_setup']/total_time*100:.1f}%)")
    service_logger.info(f"[JSON-{ann_id}] ├── Data Initialization: {timing_data['data_initialization']:.3f}s ({timing_data['data_initialization']/total_time*100:.1f}%)")
    service_logger.info(f"[JSON-{ann_id}] ├── Document Processing: {timing_data['document_processing']:.3f}s ({timing_data['document_processing']/total_time*100:.1f}%)")
    service_logger.info(f"[JSON-{ann_id}] └── JSON Saving: {timing_data['json_saving']:.3f}s ({timing_data['json_saving']/total_time*100:.1f}%)")
    service_logger.info(f"[JSON-{ann_id}] JSON annotation complete: {len(annotations)} annotations, {len(images)} images")
