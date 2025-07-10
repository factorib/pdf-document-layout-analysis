import pickle
import shutil
import time
import logging

import numpy as np
from os import makedirs
from os.path import join, exists
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_features.PdfFeatures import PdfFeatures

from bros.tokenization_bros import BrosTokenizer
from configuration import WORD_GRIDS_PATH

service_logger = logging.getLogger(__name__)

tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")


def rectangle_to_bbox(rectangle: Rectangle):
    return [rectangle.left, rectangle.top, rectangle.width, rectangle.height]


def get_words_positions(text: str, rectangle: Rectangle):
    text = text.strip()
    text_len = len(text)

    width_per_letter = rectangle.width / text_len

    words_bboxes = [Rectangle.from_coordinates(rectangle.left, rectangle.top, rectangle.left + 5, rectangle.bottom)]
    words_bboxes[-1].width = 0
    words_bboxes[-1].right = words_bboxes[-1].left

    for letter in text:
        if letter == " ":
            left = words_bboxes[-1].right + width_per_letter
            words_bboxes.append(Rectangle.from_coordinates(left, words_bboxes[-1].top, left + 5, words_bboxes[-1].bottom))
            words_bboxes[-1].width = 0
            words_bboxes[-1].right = words_bboxes[-1].left
        else:
            words_bboxes[-1].right = words_bboxes[-1].right + width_per_letter
            words_bboxes[-1].width = words_bboxes[-1].width + width_per_letter

    words = text.split()
    return words, words_bboxes


def get_subwords_positions(word: str, rectangle: Rectangle):
    width_per_letter = rectangle.width / len(word)
    word_tokens = [x.replace("#", "") for x in tokenizer.tokenize(word)]

    if not word_tokens:
        return [], []

    ids = [x[-2] for x in tokenizer(word_tokens)["input_ids"]]

    right = rectangle.left + len(word_tokens[0]) * width_per_letter
    bboxes = [Rectangle.from_coordinates(rectangle.left, rectangle.top, right, rectangle.bottom)]

    for subword in word_tokens[1:]:
        right = bboxes[-1].right + len(subword) * width_per_letter
        bboxes.append(Rectangle.from_coordinates(bboxes[-1].right, rectangle.top, right, rectangle.bottom))

    return ids, bboxes


def get_grid_words_dict(tokens: list[PdfToken]):
    texts, bbox_texts_list, inputs_ids, bbox_subword_list = [], [], [], []
    for token in tokens:
        words, words_bboxes = get_words_positions(token.content, token.bounding_box)
        texts += words
        bbox_texts_list += [rectangle_to_bbox(r) for r in words_bboxes]
        for word, word_box in zip(words, words_bboxes):
            ids, subwords_bboxes = get_subwords_positions(word, word_box)
            inputs_ids += ids
            bbox_subword_list += [rectangle_to_bbox(r) for r in subwords_bboxes]

    return {
        "input_ids": np.array(inputs_ids),
        "bbox_subword_list": np.array(bbox_subword_list),
        "texts": texts,
        "bbox_texts_list": np.array(bbox_texts_list),
    }


def create_word_grid(pdf_features_list: list[PdfFeatures]):
    # Start word grid creation timing
    start_time = time.time()
    timing_data = {}
    
    # Generate unique ID for this word grid creation process
    import uuid
    grid_id = str(uuid.uuid4())[:8]
    
    service_logger.info(f"[GRID-{grid_id}] Starting word grid creation for {len(pdf_features_list)} documents")
    
    # Stage 1: Directory Setup
    stage_start = time.time()
    makedirs(WORD_GRIDS_PATH, exist_ok=True)
    timing_data['directory_setup'] = time.time() - stage_start
    service_logger.info(f"[GRID-{grid_id}] Directory setup completed in {timing_data['directory_setup']:.3f}s")
    
    # Stage 2: Grid Processing
    stage_start = time.time()
    total_pages = 0
    pages_processed = 0
    pages_skipped = 0
    
    for pdf_features in pdf_features_list:
        for page in pdf_features.pages:
            total_pages += 1
            image_id = f"{pdf_features.file_name}_{page.page_number - 1}"
            
            # Check if grid already exists
            if exists(join(WORD_GRIDS_PATH, image_id + ".pkl")):
                pages_skipped += 1
                continue
            
            # Process page
            page_start = time.time()
            grid_words_dict = get_grid_words_dict(page.tokens)
            
            # Save grid
            with open(join(WORD_GRIDS_PATH, f"{image_id}.pkl"), mode="wb") as file:
                pickle.dump(grid_words_dict, file)
            
            pages_processed += 1
            page_time = time.time() - page_start
            
            # Log progress every 5 pages
            if pages_processed % 5 == 0:
                service_logger.info(f"[GRID-{grid_id}] Page {pages_processed} processed in {page_time:.3f}s ({len(page.tokens)} tokens)")
    
    timing_data['grid_processing'] = time.time() - stage_start
    service_logger.info(f"[GRID-{grid_id}] Grid processing completed in {timing_data['grid_processing']:.3f}s")
    
    # Calculate total time
    total_time = time.time() - start_time
    timing_data['total_time'] = total_time
    
    # Log comprehensive timing summary
    service_logger.info(f"[GRID-{grid_id}] WORD GRID TIMING SUMMARY - Total: {total_time:.3f}s")
    service_logger.info(f"[GRID-{grid_id}] ├── Directory Setup: {timing_data['directory_setup']:.3f}s ({timing_data['directory_setup']/total_time*100:.1f}%)")
    service_logger.info(f"[GRID-{grid_id}] └── Grid Processing: {timing_data['grid_processing']:.3f}s ({timing_data['grid_processing']/total_time*100:.1f}%)")
    service_logger.info(f"[GRID-{grid_id}] Word grid creation complete: {pages_processed} pages processed, {pages_skipped} pages skipped, {total_pages} total pages")


def remove_word_grids():
    shutil.rmtree(WORD_GRIDS_PATH, ignore_errors=True)
