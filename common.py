"""Shared helpers used by the pipeline, the annotation tool and the evaluator."""
import csv
import json
import os

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FACTORS = ["perspective", "lighting", "background", "document", "orientation", "quality"]


def order_corners(pts):
    """Return 4 points ordered clockwise starting at the top-left corner.

    Uses the angle around the centroid, so it works for rotated pages where
    the simple x+y / x-y trick breaks down.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]          # image y points down, so this is clockwise on screen
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def load_json(path, default=None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)               # atomic so a crash never corrupts annotations


def list_images(folder):
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTS)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
