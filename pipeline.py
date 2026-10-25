"""Classical document cleanup pipeline.

1. detect_document  - find the page's four corners (Canny edges, Otsu fallback)
2. warp_document    - perspective correction to a flat, straight-on page
3. enhance          - even out lighting by dividing out a background estimate
                      (morphological closing + median blur), then stretch contrast
"""
import cv2
import numpy as np

from common import order_corners

MIN_AREA_FRAC = 0.10    # page must cover at least 10% of the frame
MAX_AREA_FRAC = 0.97    # anything bigger is almost always the image border


def _quad_from_mask(mask):
    """Largest convex 4-sided contour in a binary mask, or None."""
    h, w = mask.shape[:2]
    img_area = float(h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        hull = cv2.convexHull(cnt)
        frac = cv2.contourArea(hull) / img_area
        if frac < MIN_AREA_FRAC:
            break
        if frac > MAX_AREA_FRAC:
            continue
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def detect_document(img, max_dim=1000):
    """Return (corners Nx2 in full-resolution pixels, method name) or (None, None)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Strategy 1: edges, thresholds set from the median intensity
    v = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    quad = _quad_from_mask(edges)
    method = "edges"

    # Strategy 2: global Otsu threshold (works when the page is brighter than the surface)
    if quad is None:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        quad = _quad_from_mask(th)
        method = "otsu"

    if quad is None:
        return None, None
    return order_corners(quad / scale), method


def warp_document(img, corners):
    tl, tr, br, bl = order_corners(corners)
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC)


def enhance(img, mode="gray"):
    """mode: 'gray' (clean grayscale) or 'binary' (black text on white)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    k = max(15, (min(gray.shape[:2]) // 40) | 1)             # odd kernel, scales with page size
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.medianBlur(background, 21)
    flat = cv2.divide(gray, background, scale=255)           # removes shadows and gradients
    lo, hi = np.percentile(flat, (1, 99))
    flat = np.clip((flat.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    if mode == "binary":
        _, flat = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return flat
