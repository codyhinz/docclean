"""Make side-by-side demo images: original (with corners drawn) | cleaned page.

    py demo_compare.py --images photos --out demo
    py demo_compare.py --images photos --out demo --gt annotations.json
    py demo_compare.py --images photos --out demo --only baseline_01.jpg light_bg_02.jpg --show

Green outline = corners the pipeline detected.
Blue outline  = hand-labeled true corners (only with --gt), plus the IoU score.
If detection fails, the panel says so, which is handy for showing the failure case.
"""
import argparse
import os

import cv2
import numpy as np

from common import list_images, load_json, order_corners
from pipeline import detect_document, enhance, warp_document

PANEL_H = 900                  # every panel is scaled to this height
GREEN, BLUE, RED = (60, 200, 60), (230, 120, 30), (40, 40, 220)
WHITE, DARK = (255, 255, 255), (35, 35, 35)


def iou(a, b, shape):
    """IoU of two quadrilaterals, rasterized at a reduced size for speed."""
    h, w = shape[:2]
    s = 1000 / max(h, w)
    size = (int(h * s) + 1, int(w * s) + 1)
    ma, mb = np.zeros(size, np.uint8), np.zeros(size, np.uint8)
    cv2.fillPoly(ma, [np.int32(np.asarray(a) * s)], 1)
    cv2.fillPoly(mb, [np.int32(np.asarray(b) * s)], 1)
    union = np.logical_or(ma, mb).sum()
    return np.logical_and(ma, mb).sum() / union if union else 0.0


def draw_quad(img, corners, color, thick):
    pts = np.int32(np.asarray(corners).reshape(4, 2))
    cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(int(v) for v in p), thick * 3, color, -1, cv2.LINE_AA)


def to_panel(img, title, color=DARK):
    """Scale to PANEL_H tall and add a title bar on top."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (max(1, int(w * PANEL_H / h)), PANEL_H), interpolation=cv2.INTER_AREA)
    bar = np.full((70, img.shape[1], 3), color, np.uint8)
    cv2.putText(bar, title, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def blank_panel(text, width=650):
    img = np.full((PANEL_H, width, 3), 235, np.uint8)
    cv2.putText(img, text, (40, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, RED, 3, cv2.LINE_AA)
    return img


def make_comparison(img, name, mode, gt):
    corners, method = detect_document(img)
    shown = img.copy()
    thick = max(3, int(max(img.shape[:2]) / 300))     # lines stay visible after shrinking

    label = name
    if name in gt and gt[name].get("corners"):
        true = order_corners(gt[name]["corners"])
        draw_quad(shown, true, BLUE, thick * 2)   # thicker so it shows under green
        if corners is not None:
            score = iou(order_corners(corners), true, img.shape)
            label += f"   IoU {score:.2f} ({'correct' if score >= 0.90 else 'miss'})"

    if corners is None:
        left = to_panel(shown, f"{name}   detection FAILED", RED)
        right = to_panel(blank_panel("No page found"), "Cleaned", RED)
        return np.hstack([left, right]), False

    draw_quad(shown, corners, GREEN, thick)
    cleaned = enhance(warp_document(img, corners), mode)
    left = to_panel(shown, label)
    right = to_panel(cleaned, f"Cleaned ({method})")
    gap = np.full((left.shape[0], 20, 3), 255, np.uint8)
    return np.hstack([left, gap, right]), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--mode", choices=["gray", "binary"], default="gray")
    ap.add_argument("--gt", help="annotations.json to also draw the true corners and IoU")
    ap.add_argument("--only", nargs="+", help="just these file names")
    ap.add_argument("--show", action="store_true", help="open each result in a window (any key = next)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt = load_json(args.gt) if args.gt else {}
    names = args.only or list_images(args.images)

    for name in names:
        img = cv2.imread(os.path.join(args.images, name))
        if img is None:
            print(f"  could not read {name}, skipping")
            continue
        out, ok = make_comparison(img, name, args.mode, gt)
        path = os.path.join(args.out, os.path.splitext(name)[0] + "_compare.jpg")
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {name:40s} {'ok' if ok else 'FAILED'}  ->  {path}")

        if args.show:
            win = "demo (any key = next, Esc = quit)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, out)
            if cv2.waitKey(0) == 27:
                break
    cv2.destroyAllWindows() if args.show else None


if __name__ == "__main__":
    main()