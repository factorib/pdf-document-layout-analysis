import time
from fast_trainer.PdfSegment import PdfSegment
from pdf_features.PdfPage import PdfPage
from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType

from data_model.PdfImages import PdfImages
from configuration import service_logger

try:
    from rtree import index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    print("Warning: rtree not available, falling back to O(n²) algorithm")

import numpy as np
from functools import lru_cache

# Global cache for bounding box calculations to avoid repeated computations
_bbox_cache = {}


@lru_cache(maxsize=512)
def get_cached_center(left: float, top: float, right: float, bottom: float):
    """Cache center calculations for frequently accessed bounding boxes."""
    return ((left + right) / 2, (top + bottom) / 2)


def clear_bbox_cache():
    """Clear the bounding box cache to prevent memory leaks."""
    global _bbox_cache
    _bbox_cache.clear()


def find_segment_for_token(token: PdfToken, segments: list[PdfSegment], tokens_by_segments, spatial_index=None):
    best_score: float = 0
    most_probable_segment: PdfSegment | None = None
    
    if spatial_index is not None and RTREE_AVAILABLE:
        # Use spatial indexing for O(log n) performance
        token_bounds = (
            token.bounding_box.left,
            token.bounding_box.top,
            token.bounding_box.right,
            token.bounding_box.bottom
        )
        candidate_indices = list(spatial_index.intersection(token_bounds))
        candidates = [segments[i] for i in candidate_indices]
        
        # Early termination if no spatial candidates found
        if not candidates:
            return
    else:
        # Fallback to O(n) brute force
        candidates = segments
    
    # Early termination optimization: sort candidates by area overlap potential
    if len(candidates) > 3:
        # Quick distance-based pre-filter for large candidate sets using cached centers
        token_center_x, token_center_y = get_cached_center(
            token.bounding_box.left, token.bounding_box.top,
            token.bounding_box.right, token.bounding_box.bottom
        )
        
        def candidate_priority(segment):
            seg_center_x, seg_center_y = get_cached_center(
                segment.bounding_box.left, segment.bounding_box.top,
                segment.bounding_box.right, segment.bounding_box.bottom
            )
            return (token_center_x - seg_center_x) ** 2 + (token_center_y - seg_center_y) ** 2
        
        # Process closest candidates first for better early termination
        candidates = sorted(candidates, key=candidate_priority)[:min(len(candidates), 8)]
    
    for segment in candidates:
        intersection_percentage = token.bounding_box.get_intersection_percentage(segment.bounding_box)
        if intersection_percentage > best_score:
            best_score = intersection_percentage
            most_probable_segment = segment
            # Early termination with perfect or near-perfect match
            if best_score >= 99:
                break
            # Early termination with very good match if we have many candidates
            elif best_score >= 90 and len(candidates) > 5:
                break
    
    if most_probable_segment:
        tokens_by_segments.setdefault(most_probable_segment, list()).append(token)


def get_average_reading_order_for_segment(page: PdfPage, tokens_for_segment: list[PdfToken], token_to_order_map=None):
    if token_to_order_map is not None:
        # Use pre-computed mapping for O(1) lookup per token ID
        reading_order_sum: int = sum(token_to_order_map[token.id] for token in tokens_for_segment)
    else:
        # Fallback to O(n) lookup per token
        reading_order_sum: int = sum(page.tokens.index(token) for token in tokens_for_segment)
    
    return reading_order_sum / len(tokens_for_segment)


def get_distance_between_segments(segment1: PdfSegment, segment2: PdfSegment):
    center_1_x = (segment1.bounding_box.left + segment1.bounding_box.right) / 2
    center_1_y = (segment1.bounding_box.top + segment1.bounding_box.bottom) / 2
    center_2_x = (segment2.bounding_box.left + segment2.bounding_box.right) / 2
    center_2_y = (segment2.bounding_box.top + segment2.bounding_box.bottom) / 2
    return ((center_1_x - center_2_x) ** 2 + (center_1_y - center_2_y) ** 2) ** 0.5


def get_segment_centers_vectorized(segments):
    """Pre-compute all segment centers using vectorization for efficiency."""
    centers = np.array([
        [(seg.bounding_box.left + seg.bounding_box.right) / 2,
         (seg.bounding_box.top + seg.bounding_box.bottom) / 2]
        for seg in segments
    ])
    return centers


def find_closest_segment_vectorized(target_segment, segments, segment_centers=None):
    """Find closest segment using vectorized distance calculation."""
    if not segments:
        return None
        
    if segment_centers is None:
        segment_centers = get_segment_centers_vectorized(segments)
    
    target_center = np.array([
        (target_segment.bounding_box.left + target_segment.bounding_box.right) / 2,
        (target_segment.bounding_box.top + target_segment.bounding_box.bottom) / 2
    ])
    
    # Vectorized distance calculation
    distances = np.linalg.norm(segment_centers - target_center, axis=1)
    closest_index = np.argmin(distances)
    return segments[closest_index], closest_index


def add_no_token_segments(segments, no_token_segments):
    if not no_token_segments:
        return
        
    if segments:
        # Pre-compute centers once for all segments
        segment_centers = get_segment_centers_vectorized(segments)
        
        # Process no-token segments in batches for better cache locality
        for no_token_segment in no_token_segments:
            closest_segment, closest_index = find_closest_segment_vectorized(
                no_token_segment, segments, segment_centers
            )
            
            # Insert based on vertical position
            if closest_segment.bounding_box.top < no_token_segment.bounding_box.top:
                segments.insert(closest_index + 1, no_token_segment)
                # Update centers array after insertion
                segment_centers = np.insert(segment_centers, closest_index + 1, 
                    [(no_token_segment.bounding_box.left + no_token_segment.bounding_box.right) / 2,
                     (no_token_segment.bounding_box.top + no_token_segment.bounding_box.bottom) / 2], axis=0)
            else:
                segments.insert(closest_index, no_token_segment)
                # Update centers array after insertion
                segment_centers = np.insert(segment_centers, closest_index,
                    [(no_token_segment.bounding_box.left + no_token_segment.bounding_box.right) / 2,
                     (no_token_segment.bounding_box.top + no_token_segment.bounding_box.bottom) / 2], axis=0)
    else:
        # Efficient sorting for empty segments list
        segments.extend(sorted(no_token_segments, key=lambda r: (r.bounding_box.left, r.bounding_box.top)))


def filter_and_sort_segments(page, tokens_by_segments, types, token_to_order_map=None):
    # Use set for O(1) lookup instead of list inclusion check
    types_set = set(types) if not isinstance(types, set) else types
    
    # Filter segments more efficiently
    filtered_segments = [seg for seg in tokens_by_segments.keys() if seg.segment_type in types_set]
    
    if not filtered_segments:
        return []
    
    # Pre-compute reading orders for all filtered segments
    order = {seg: get_average_reading_order_for_segment(page, tokens_by_segments[seg], token_to_order_map) 
             for seg in filtered_segments}
    
    # Sort using pre-computed orders
    return sorted(filtered_segments, key=lambda seg: order[seg])


def get_ordered_segments_for_page(segments_for_page: list[PdfSegment], page: PdfPage):
    # Build spatial index if available - lower threshold for better coverage
    spatial_index = None
    if RTREE_AVAILABLE and len(segments_for_page) > 5:  # Lower threshold for more aggressive optimization
        spatial_index = index.Index()
        for i, segment in enumerate(segments_for_page):
            spatial_index.insert(i, (
                segment.bounding_box.left,
                segment.bounding_box.top,
                segment.bounding_box.right,
                segment.bounding_box.bottom
            ))
    
    # Pre-compute token-to-order mapping for O(1) lookup using token IDs
    token_to_order_map = {token.id: i for i, token in enumerate(page.tokens)}
    
    tokens_by_segments: dict[PdfSegment, list[PdfToken]] = {}
    for token in page.tokens:
        find_segment_for_token(token, segments_for_page, tokens_by_segments, spatial_index)

    page_number_segment: None | PdfSegment = None
    if tokens_by_segments:
        last_segment = max(tokens_by_segments.keys(), key=lambda seg: seg.bounding_box.top)
        if last_segment.text_content and len(last_segment.text_content) < 5:
            page_number_segment = last_segment
            del tokens_by_segments[last_segment]

    header_segments: list[PdfSegment] = filter_and_sort_segments(page, tokens_by_segments, {TokenType.PAGE_HEADER}, token_to_order_map)
    paragraph_types = {t for t in TokenType if t.name not in {"PAGE_HEADER", "PAGE_FOOTER", "FOOTNOTE"}}
    paragraph_segments = filter_and_sort_segments(page, tokens_by_segments, paragraph_types, token_to_order_map)
    footer_segments = filter_and_sort_segments(page, tokens_by_segments, {TokenType.PAGE_FOOTER, TokenType.FOOTNOTE}, token_to_order_map)
    if page_number_segment:
        footer_segments.append(page_number_segment)
    ordered_segments = header_segments + paragraph_segments + footer_segments
    no_token_segments = [segment for segment in segments_for_page if segment not in ordered_segments]
    add_no_token_segments(ordered_segments, no_token_segments)
    return ordered_segments


def get_reading_orders(pdf_images_list: list[PdfImages], predicted_segments: list[PdfSegment]):
    start_time = time.time()
    ordered_segments: list[PdfSegment] = []
    
    service_logger.info(f"Reading order analysis: Processing {len(predicted_segments)} segments across {len(pdf_images_list)} documents")
    
    try:
        for pdf_idx, pdf_images in enumerate(pdf_images_list):
            pdf_start = time.time()
            pdf_name = pdf_images.pdf_features.file_name
            segments_for_file = [segment for segment in predicted_segments if segment.pdf_name == pdf_name]
            
            for page_idx, page in enumerate(pdf_images.pdf_features.pages):
                page_start = time.time()
                segments_for_page = [segment for segment in segments_for_file if segment.page_number == page.page_number]
                page_ordered = get_ordered_segments_for_page(segments_for_page, page)
                ordered_segments.extend(page_ordered)
                page_time = time.time() - page_start
                
                if len(segments_for_page) > 0:  # Only log pages with segments
                    service_logger.info(f"Reading order analysis: Page {page.page_number} processed in {page_time:.3f}s ({len(segments_for_page)} segments → {len(page_ordered)} ordered)")
            
            pdf_time = time.time() - pdf_start
            service_logger.info(f"Reading order analysis: Document {pdf_idx + 1} processed in {pdf_time:.3f}s")
            
        total_time = time.time() - start_time
        service_logger.info(f"Reading order analysis: SUMMARY - Total:{total_time:.3f}s, Input:{len(predicted_segments)} segments, Output:{len(ordered_segments)} ordered segments")
        return ordered_segments
    finally:
        # Clear cache to prevent memory leaks
        clear_bbox_cache()
