import json
import pickle
from os.path import join
from pathlib import Path
from statistics import mode
import time
import logging

from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
from data_model.PdfImages import PdfImages
from configuration import ROOT_PATH, DOCLAYNET_TYPE_BY_ID
from data_model.Prediction import Prediction

service_logger = logging.getLogger(__name__)


def get_prediction_from_annotation(annotation, images_names, vgt_predictions_dict):
    pdf_name = images_names[annotation["image_id"]][:-4]
    category_id = annotation["category_id"]
    bounding_box = Rectangle.from_width_height(
        left=int(annotation["bbox"][0]),
        top=int(annotation["bbox"][1]),
        width=int(annotation["bbox"][2]),
        height=int(annotation["bbox"][3]),
    )

    prediction = Prediction(
        bounding_box=bounding_box, category_id=category_id, score=round(float(annotation["score"]) * 100, 2)
    )
    vgt_predictions_dict.setdefault(pdf_name, list()).append(prediction)


def get_vgt_predictions(model_name: str) -> dict[str, list[Prediction]]:
    output_dir: str = f"model_output_{model_name}"
    model_output_json_path = join(str(ROOT_PATH), output_dir, "inference", "coco_instances_results.json")
    annotations = json.loads(Path(model_output_json_path).read_text())

    test_json_path = join(str(ROOT_PATH), "jsons", "test.json")
    coco_truth = json.loads(Path(test_json_path).read_text())

    images_names = {value["id"]: value["file_name"] for value in coco_truth["images"]}

    vgt_predictions_dict = dict()
    for annotation in annotations:
        get_prediction_from_annotation(annotation, images_names, vgt_predictions_dict)

    return vgt_predictions_dict


def find_best_prediction_for_token(page_pdf_name, token, vgt_predictions_dict, most_probable_tokens_by_predictions):
    best_score: float = 0
    most_probable_prediction: Prediction | None = None
    for prediction in vgt_predictions_dict[page_pdf_name]:
        if prediction.score > best_score and prediction.bounding_box.get_intersection_percentage(token.bounding_box):
            best_score = prediction.score
            most_probable_prediction = prediction
            if best_score >= 99:
                break
    if most_probable_prediction:
        most_probable_tokens_by_predictions.setdefault(most_probable_prediction, list()).append(token)
    else:
        dummy_prediction = Prediction(bounding_box=token.bounding_box, category_id=10, score=0.0)
        most_probable_tokens_by_predictions.setdefault(dummy_prediction, list()).append(token)


def get_merged_prediction_type(to_merge: list[Prediction]):
    table_exists = any([p.category_id == 9 for p in to_merge])
    if not table_exists:
        return mode([p.category_id for p in sorted(to_merge, key=lambda x: -x.score)])
    return 9


def merge_colliding_predictions(predictions: list[Prediction]):
    predictions = [p for p in predictions if not p.score < 20]
    while True:
        new_predictions, merged = [], False
        while predictions:
            p1 = predictions.pop(0)
            to_merge = [p for p in predictions if p1.bounding_box.get_intersection_percentage(p.bounding_box) > 0]
            for prediction in to_merge:
                predictions.remove(prediction)
            if to_merge:
                to_merge.append(p1)
                p1.bounding_box = Rectangle.merge_rectangles([prediction.bounding_box for prediction in to_merge])
                p1.category_id = get_merged_prediction_type(to_merge)
                merged = True
            new_predictions.append(p1)
        if not merged:
            return new_predictions
        predictions = new_predictions


def get_pdf_segments_for_page(page, pdf_name, page_pdf_name, vgt_predictions_dict):
    most_probable_pdf_segments_for_page: list[PdfSegment] = []
    most_probable_tokens_by_predictions: dict[Prediction, list[PdfToken]] = {}
    vgt_predictions_dict[page_pdf_name] = merge_colliding_predictions(vgt_predictions_dict[page_pdf_name])

    for token in page.tokens:
        find_best_prediction_for_token(page_pdf_name, token, vgt_predictions_dict, most_probable_tokens_by_predictions)

    for prediction, tokens in most_probable_tokens_by_predictions.items():
        new_segment = PdfSegment.from_pdf_tokens(tokens, pdf_name)
        new_segment.bounding_box = prediction.bounding_box
        new_segment.segment_type = TokenType.from_text(DOCLAYNET_TYPE_BY_ID[prediction.category_id])
        most_probable_pdf_segments_for_page.append(new_segment)

    no_token_predictions = [
        prediction
        for prediction in vgt_predictions_dict[page_pdf_name]
        if prediction not in most_probable_tokens_by_predictions
    ]

    for prediction in no_token_predictions:
        segment_type = TokenType.from_text(DOCLAYNET_TYPE_BY_ID[prediction.category_id])
        page_number = page.page_number
        new_segment = PdfSegment(page_number, prediction.bounding_box, "", segment_type, pdf_name)
        most_probable_pdf_segments_for_page.append(new_segment)

    return most_probable_pdf_segments_for_page


def prediction_exists_for_page(page_pdf_name, vgt_predictions_dict):
    return page_pdf_name in vgt_predictions_dict


def get_most_probable_pdf_segments(model_name: str, pdf_images_list: list[PdfImages], save_output: bool = False):
    # Start segment extraction timing
    start_time = time.time()
    timing_data = {}
    
    # Generate unique ID for this segment extraction process
    import uuid
    seg_id = str(uuid.uuid4())[:8]
    
    service_logger.info(f"[SEG-{seg_id}] Starting segment extraction for model: {model_name}")
    
    # Stage 1: Initialization and VGT Predictions Loading
    stage_start = time.time()
    most_probable_pdf_segments: list[PdfSegment] = []
    vgt_predictions_dict = get_vgt_predictions(model_name)
    timing_data['vgt_predictions_loading'] = time.time() - stage_start
    service_logger.info(f"[SEG-{seg_id}] VGT predictions loaded in {timing_data['vgt_predictions_loading']:.3f}s")
    
    # Stage 2: PDF Features Processing
    stage_start = time.time()
    pdf_features_list: list[PdfFeatures] = [pdf_images.pdf_features for pdf_images in pdf_images_list]
    timing_data['pdf_features_processing'] = time.time() - stage_start
    service_logger.info(f"[SEG-{seg_id}] PDF features processed in {timing_data['pdf_features_processing']:.3f}s")
    
    # Stage 3: Page-by-Page Segment Extraction
    stage_start = time.time()
    total_pages = 0
    pages_processed = 0
    
    for pdf_features in pdf_features_list:
        for page in pdf_features.pages:
            total_pages += 1
            page_pdf_name = pdf_features.file_name + "_" + str(page.page_number - 1)
            if not prediction_exists_for_page(page_pdf_name, vgt_predictions_dict):
                continue
            
            page_start = time.time()
            page_segments = get_pdf_segments_for_page(page, pdf_features.file_name, page_pdf_name, vgt_predictions_dict)
            most_probable_pdf_segments.extend(page_segments)
            pages_processed += 1
            page_time = time.time() - page_start
            
            if pages_processed % 5 == 0:  # Log every 5 pages
                service_logger.info(f"[SEG-{seg_id}] Page {pages_processed} processed in {page_time:.3f}s ({len(page_segments)} segments)")
    
    timing_data['page_segment_extraction'] = time.time() - stage_start
    service_logger.info(f"[SEG-{seg_id}] Page segment extraction completed in {timing_data['page_segment_extraction']:.3f}s ({pages_processed}/{total_pages} pages)")
    
    # Stage 4: Output Saving (if requested)
    if save_output:
        stage_start = time.time()
        save_path = join(ROOT_PATH, f"model_output_{model_name}", "predicted_segments.pickle")
        with open(save_path, mode="wb") as file:
            pickle.dump(most_probable_pdf_segments, file)
        timing_data['output_saving'] = time.time() - stage_start
        service_logger.info(f"[SEG-{seg_id}] Output saving completed in {timing_data['output_saving']:.3f}s")
    
    # Calculate total time
    total_time = time.time() - start_time
    timing_data['total_time'] = total_time
    
    # Log comprehensive timing summary
    service_logger.info(f"[SEG-{seg_id}] SEGMENT EXTRACTION TIMING SUMMARY - Total: {total_time:.3f}s")
    service_logger.info(f"[SEG-{seg_id}] ├── VGT Predictions Loading: {timing_data['vgt_predictions_loading']:.3f}s ({timing_data['vgt_predictions_loading']/total_time*100:.1f}%)")
    service_logger.info(f"[SEG-{seg_id}] ├── PDF Features Processing: {timing_data['pdf_features_processing']:.3f}s ({timing_data['pdf_features_processing']/total_time*100:.1f}%)")
    service_logger.info(f"[SEG-{seg_id}] ├── Page Segment Extraction: {timing_data['page_segment_extraction']:.3f}s ({timing_data['page_segment_extraction']/total_time*100:.1f}%)")
    if save_output:
        service_logger.info(f"[SEG-{seg_id}] └── Output Saving: {timing_data['output_saving']:.3f}s ({timing_data['output_saving']/total_time*100:.1f}%)")
    else:
        service_logger.info(f"[SEG-{seg_id}] └── Output Saving: Skipped")
    service_logger.info(f"[SEG-{seg_id}] Segment extraction complete: {len(most_probable_pdf_segments)} segments found")
    
    return most_probable_pdf_segments
