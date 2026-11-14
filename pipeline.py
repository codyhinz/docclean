"""Classical document cleanup pipeline.

1. detect_document  - find the page's four corners. Builds candidate outlines from
                      several cues (edges on brightness and on each Lab color channel,
                      thresholded brightness and color), rejects shapes that touch the
                      frame edge, then keeps the one that differs most from its
                      surroundings and whose outline follows real edges. An outline
                      that encloses the pick is preferred, so photos inside a page lose.
2. warp_document    - perspective correction to a flat, straight-on page
3. enhance          - even out lighting by dividing out a background estimate
                      (morphological closing + median blur), then stretch contrast
"""
import cv2
import numpy as np

from common import order_corners

WORK_DIM = 800            # detection runs on a copy this many pixels on its long side
MIN_AREA_FRAC = 0.01      # page must cover at least 1% of the frame
MAX_AREA_FRAC = 0.90      # bigger than this is almost always the table or the frame itself
NEST_KEEP = 0.35         # an enclosing outline wins if it scores at least this share of the best
BORDER_FRAC = 0.005       # a corner closer than this to the frame edge means the shape is cut off


def _quads_from_mask(mask, out):
    """Every convex 4-sided outline in a binary mask, inner contours included."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    area_img = float(h * w)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        a = cv2.contourArea(hull) / area_img
        if a < MIN_AREA_FRAC or a > MAX_AREA_FRAC:
            continue
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                out.append(approx.reshape(4, 2).astype(np.float32))
                break


def _edge_support(q, edges):
    """Share of points along the outline that sit on a real edge."""
    pts = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        for t in np.linspace(0, 1, 30, endpoint=False):
            pts.append(a + t * (b - a))
    pts = np.round(np.array(pts)).astype(int)
    h, w = edges.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return float(np.mean(edges[pts[:, 1], pts[:, 0]] > 0))


def _score(quad, lab, edges):
    """How much the quad looks like a sheet of paper lying on a surface."""
    h, w = edges.shape
    m = BORDER_FRAC * max(h, w)
    if np.any(quad[:, 0] < m) or np.any(quad[:, 0] > w - 1 - m) or \
       np.any(quad[:, 1] < m) or np.any(quad[:, 1] > h - 1 - m):
        return -1.0                                  # touches the frame edge: cut off or not a page
    q = order_corners(quad)
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 1e-3 or max(sides) / min(sides) > 4.0:
        return -1.0                                  # too thin to be a sheet of paper
    inner = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inner, [q.astype(np.int32)], 255)
    k = max(3, int(0.02 * max(h, w)))
    ring = cv2.dilate(inner, np.ones((k, k), np.uint8)) & ~inner
    core = cv2.erode(inner, np.ones((k, k), np.uint8))
    if cv2.countNonZero(core) == 0 or cv2.countNonZero(ring) == 0:
        return -1.0
    Lin, ain, bin_ = cv2.mean(lab, core)[:3]
    Lout, aout, bout = cv2.mean(lab, ring)[:3]
    dL = Lin - Lout
    dc = np.hypot(ain - aout, bin_ - bout)
    # A text block inside a page is darker than its white margin and the same color as it.
    # A page can be darker than a bright table, but then it differs in color (paper vs. wood).
    if dL < -5 and dc < 4:
        return -1.0
    contrast = 0.5 * abs(dL) + max(dL, 0.0) + 1.5 * dc
    support = _edge_support(q, edges)
    area = cv2.countNonZero(inner) / float(h * w)
    return contrast * (0.2 + support) * (1.0 + 0.5 * np.sqrt(area))


def detect_document(img, max_dim=WORK_DIM):
    """Return (corners Nx2 in full-resolution pixels, method name) or (None, None).

    Builds candidate page outlines from several cues (edges on brightness and on each
    color channel, plus thresholded brightness and color), then keeps the candidate that
    differs most from its surroundings and whose outline follows real edges.
    """
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    chans = cv2.split(lab)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k5 = np.ones((5, 5), np.uint8)

    cands = {"edges": [], "threshold": []}

    # Cue 1: edges on brightness and on each Lab channel
    edge_any = np.zeros(gray.shape, np.uint8)
    for chan, levels in ((gray, ((20, 60), (40, 120))),
                         (chans[1], ((4, 12), (8, 24))),
                         (chans[2], ((4, 12), (8, 24)))):
        for lo, hi in levels:
            e = cv2.Canny(chan, lo, hi)
            edge_any |= e
            _quads_from_mask(cv2.morphologyEx(e, cv2.MORPH_CLOSE, k5, iterations=2), cands["edges"])
    v = float(np.median(gray))
    e = cv2.Canny(gray, int(0.66 * v), int(min(255, 1.33 * v)))
    _quads_from_mask(cv2.morphologyEx(e, cv2.MORPH_CLOSE, k5, iterations=2), cands["edges"])

    # Cue 2: thresholds on brightness and on the blue-yellow channel, both directions
    for chan in (chans[0], chans[2]):
        _, th = cv2.threshold(chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for mk in (th, 255 - th):
            mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, k5)
            mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            _quads_from_mask(mk, cands["threshold"])

    edges = cv2.dilate(edge_any, np.ones((3, 3), np.uint8))
    scored = []
    for method, quads in cands.items():
        for q in quads:
            sc = _score(q, lab, edges)
            if sc > 0:
                scored.append((sc, cv2.contourArea(order_corners(q)), q, method))
    if not scored:
        return None, None
    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    # A photo or headline inside a page is also a high-contrast rectangle. If a reasonably
    # strong candidate fully encloses the current pick, the page is the enclosing one.
    changed = True
    while changed:
        changed = False
        for cand in scored:
            if cand[1] > 1.5 * best[1] and cand[0] >= NEST_KEEP * best[0] and _contains(cand[2], best[2]):
                best, changed = cand, True
                break
    return order_corners(best[2] / scale), best[3]


def _contains(outer, inner, tol=3.0):
    poly = order_corners(outer).reshape(-1, 1, 2)
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -tol for x, y in inner)


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