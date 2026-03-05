"""
Image Processing Pipeline — Stage 5 Backend
Implements: BilateralFilter → AdaptiveThreshold → Canny → MorphClose → Dilate → Binary Threshold
             → connectedComponents → Region Merge → findContours (RETR_CCOMP) → approxPolyDP
             → Deterministic ID Sort → Centroid Calculation → SVG Path Export
"""

import time
import cv2
import numpy as np
from typing import Any

# ─── Constants ─────────────────────────────────────────────────────────────────
MAX_DIMENSION = 1024
MIN_REGION_AREA_RATIO = 0.0005  # 0.05% of total image area
MAX_REGIONS = 1000
EDGE_DENSITY_THRESHOLD = 0.35   # 35% — reject or increase smoothing
NUM_PALETTE_COLORS = 10


def process_image(image_bytes: bytes, original_width: int, original_height: int) -> dict[str, Any]:
    """
    Full processing pipeline. Returns the structured JSON contract.
    """
    start_time = time.time()

    # ── 1. Decode & Resize ─────────────────────────────────────────────────────
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    h, w = img.shape[:2]
    scale_factor = min(MAX_DIMENSION / w, MAX_DIMENSION / h, 1.0)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ── 2. Edge Detection Pipeline ─────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # BilateralFilter — preserve edges while smoothing
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # AdaptiveThreshold — local contrast-aware binarization
    adaptive = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )

    # Canny edge detection
    edges = cv2.Canny(filtered, 50, 150)

    # Combine adaptive threshold + canny for stronger boundaries
    combined_edges = cv2.bitwise_or(adaptive, edges)

    # Morphological Close — seal micro-gaps
    kernel_close = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_close)

    # Dilate — thicken boundaries slightly
    kernel_dilate = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

    # ── Strict Binary Threshold (0/255) — eliminate anti-alias artifacts ───────
    _, binary_edges = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)

    # ── Complexity Guard ───────────────────────────────────────────────────────
    edge_density = np.count_nonzero(binary_edges) / binary_edges.size
    if edge_density > EDGE_DENSITY_THRESHOLD:
        # Increase smoothing and retry
        filtered = cv2.bilateralFilter(gray, d=15, sigmaColor=120, sigmaSpace=120)
        edges = cv2.Canny(filtered, 80, 200)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
        _, binary_edges = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)

    # ── 3. Flood Fill / Region Extraction ──────────────────────────────────────
    # Invert: edges become 0 (barriers), fill regions become 255
    inverted = cv2.bitwise_not(binary_edges)

    num_labels, labels = cv2.connectedComponents(inverted, connectivity=8)

    # Ignore label 0 (background)
    total_pixels = new_w * new_h
    min_area = int(total_pixels * MIN_REGION_AREA_RATIO)

    # ── 4. Region Filtering & Merging ──────────────────────────────────────────
    region_areas: dict[int, int] = {}
    for label_id in range(1, num_labels):
        area = int(np.sum(labels == label_id))
        region_areas[label_id] = area

    # Identify small regions to merge
    small_regions = {lid for lid, area in region_areas.items() if area < min_area}
    large_regions = {lid for lid, area in region_areas.items() if area >= min_area}

    # Merge small regions into their largest adjacent neighbor
    if small_regions and large_regions:
        for small_id in small_regions:
            mask = np.uint8(labels == small_id) * 255
            dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            overlap = dilated_mask & (labels > 0).astype(np.uint8) * 255

            # Find neighboring labels
            neighbor_labels = set(np.unique(labels[overlap > 0])) - {0, small_id} - small_regions
            if neighbor_labels:
                # Pick the largest neighbor
                best_neighbor = max(neighbor_labels, key=lambda lid: region_areas.get(lid, 0))
                labels[labels == small_id] = best_neighbor
                region_areas[best_neighbor] = region_areas.get(best_neighbor, 0) + region_areas[small_id]

    # Rebuild valid label set after merging
    valid_labels = sorted(set(np.unique(labels)) - {0})

    # ── Safety Cap ─────────────────────────────────────────────────────────────
    if len(valid_labels) > MAX_REGIONS:
        # Sort by area ascending, merge smallest until under cap
        label_areas = [(lid, int(np.sum(labels == lid))) for lid in valid_labels]
        label_areas.sort(key=lambda x: x[1])

        while len(label_areas) > MAX_REGIONS:
            smallest_id, _ = label_areas.pop(0)
            # Merge into the next smallest
            if label_areas:
                target_id = label_areas[0][0]
                labels[labels == smallest_id] = target_id
                label_areas[0] = (target_id, label_areas[0][1] + _)

        valid_labels = [la[0] for la in label_areas]

    # ── 5. Contour Extraction with Hierarchy ───────────────────────────────────
    raw_regions: list[dict[str, Any]] = []

    for label_id in valid_labels:
        mask = np.uint8(labels == label_id) * 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Use the largest contour for this region
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Convert to SVG path string
        path = _contour_to_svg_path(approx)

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(approx)

        # Centroid via moments
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = bx + bw // 2
            cy = by + bh // 2

        raw_regions.append({
            "label_id": label_id,
            "path": path,
            "boundingBox": {"x": bx, "y": by, "w": bw, "h": bh},
            "centroid": {"x": cx, "y": cy},
            "area": int(area),
        })

    # ── 6. Deterministic Sorting (Y asc, X asc) → Sequential IDs ──────────────
    raw_regions.sort(key=lambda r: (r["boundingBox"]["y"], r["boundingBox"]["x"]))

    # ── 7. Color Quantization (K-Means Palette) ───────────────────────────────
    palette = _extract_palette(img, NUM_PALETTE_COLORS)

    # Assign each region a color index based on dominant color inside its mask
    regions_out: list[dict[str, Any]] = []
    parent_map: dict[int, int | None] = {}  # label_id → assigned sequential id
    children_map: dict[int, list[int]] = {}

    for seq_id, region in enumerate(raw_regions, start=1):
        # Determine dominant color inside the region
        mask = np.uint8(labels == region["label_id"]) * 255
        color_index = _assign_color_index(img, mask, palette)

        regions_out.append({
            "id": seq_id,
            "parentId": None,
            "children": [],
            "colorIndex": color_index,
            "path": region["path"],
            "boundingBox": region["boundingBox"],
            "centroid": region["centroid"],
        })

    # ── Build Hierarchy (simple containment check via bounding boxes) ──────────
    for i, outer in enumerate(regions_out):
        for j, inner in enumerate(regions_out):
            if i == j:
                continue
            ob = outer["boundingBox"]
            ib = inner["boundingBox"]
            # Check if inner is fully contained within outer
            if (ib["x"] >= ob["x"] and ib["y"] >= ob["y"] and
                ib["x"] + ib["w"] <= ob["x"] + ob["w"] and
                ib["y"] + ib["h"] <= ob["y"] + ob["h"] and
                outer["id"] != inner.get("parentId")):
                # Only set parent if inner doesn't already have a closer parent
                if inner["parentId"] is None:
                    inner["parentId"] = outer["id"]
                    outer["children"].append(inner["id"])
                elif _bbox_area(ob) < _bbox_area(
                    next(r["boundingBox"] for r in regions_out if r["id"] == inner["parentId"])
                ):
                    # This outer is smaller (closer parent) — re-assign
                    old_parent = next(r for r in regions_out if r["id"] == inner["parentId"])
                    old_parent["children"].remove(inner["id"])
                    inner["parentId"] = outer["id"]
                    outer["children"].append(inner["id"])

    # ── 8. Outline Path (union of all edge contours) ──────────────────────────
    outline_contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline_path = ""
    for c in outline_contours:
        if cv2.contourArea(c) > min_area:
            eps = 0.001 * cv2.arcLength(c, True)
            approx_c = cv2.approxPolyDP(c, eps, True)
            outline_path += _contour_to_svg_path(approx_c) + " "
    outline_path = outline_path.strip()

    # ── 9. Build Response ──────────────────────────────────────────────────────
    processing_time_ms = int((time.time() - start_time) * 1000)

    return {
        "meta": {
            "originalWidth": original_width,
            "originalHeight": original_height,
            "resizedWidth": new_w,
            "resizedHeight": new_h,
            "regionCount": len(regions_out),
            "processingTimeMs": processing_time_ms,
        },
        "width": new_w,
        "height": new_h,
        "outlinePath": outline_path if outline_path else None,
        "regions": regions_out,
        "palette": [_bgr_to_hex(c) for c in palette],
    }


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _contour_to_svg_path(contour: np.ndarray) -> str:
    """Convert an OpenCV contour to an SVG path string."""
    points = contour.reshape(-1, 2)
    parts = [f"M {points[0][0]} {points[0][1]}"]
    for p in points[1:]:
        parts.append(f"L {p[0]} {p[1]}")
    parts.append("Z")
    return " ".join(parts)


def _extract_palette(img: np.ndarray, k: int) -> list[np.ndarray]:
    """K-Means color quantization to generate a palette."""
    pixels = img.reshape(-1, 3).astype(np.float32)

    # Subsample for speed
    if len(pixels) > 50000:
        indices = np.random.choice(len(pixels), 50000, replace=False)
        pixels = pixels[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    return [c.astype(int) for c in centers]


def _assign_color_index(img: np.ndarray, mask: np.ndarray, palette: list[np.ndarray]) -> int:
    """Find which palette color is dominant inside the masked region."""
    region_pixels = img[mask > 0]
    if len(region_pixels) == 0:
        return 0

    avg_color = np.mean(region_pixels, axis=0)

    # Find nearest palette color (Euclidean distance)
    min_dist = float("inf")
    best_idx = 0
    for i, pc in enumerate(palette):
        dist = np.linalg.norm(avg_color - pc)
        if dist < min_dist:
            min_dist = dist
            best_idx = i

    return best_idx


def _bgr_to_hex(bgr: np.ndarray) -> str:
    """Convert BGR numpy array to hex color string."""
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02X}{g:02X}{b:02X}"


def _bbox_area(bb: dict) -> int:
    return bb["w"] * bb["h"]
