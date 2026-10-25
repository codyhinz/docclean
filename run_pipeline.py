"""Run the pipeline over a folder of photos.

    python run_pipeline.py --images photos --out results

Writes results/predictions.json (automatic detections only, used by evaluate.py)
and results/cleaned/<name>.png for every image it could process.

--manual annotations.json uses hand-marked corners for photos where automatic
detection failed (the manual fallback from the proposal). Those images are
still recorded as failures in predictions.json so the evaluation stays honest.
"""
import argparse
import os

import cv2

from common import list_images, load_json, save_json
from pipeline import detect_document, enhance, warp_document


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--mode", choices=["gray", "binary"], default="gray")
    ap.add_argument("--manual", help="annotations.json to use as fallback corners")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "cleaned"), exist_ok=True)
    manual = load_json(args.manual) if args.manual else {}
    preds = {}

    for name in list_images(args.images):
        img = cv2.imread(os.path.join(args.images, name))   # applies EXIF rotation
        if img is None:
            print(f"  could not read {name}, skipping")
            continue
        h, w = img.shape[:2]
        corners, method = detect_document(img)
        preds[name] = {"width": w, "height": h, "method": method,
                       "corners": None if corners is None else corners.round(1).tolist()}

        used = corners
        if used is None and name in manual and manual[name].get("corners"):
            used, method = manual[name]["corners"], "manual"
        status = method or "FAILED"
        print(f"  {name:40s} {status}")
        if used is not None:
            out = enhance(warp_document(img, used), args.mode)
            cv2.imwrite(os.path.join(args.out, "cleaned", os.path.splitext(name)[0] + ".png"), out)

    save_json(os.path.join(args.out, "predictions.json"), preds)
    found = sum(p["corners"] is not None for p in preds.values())
    print(f"\nAutomatic detection returned corners for {found}/{len(preds)} images.")
    print(f"Predictions saved to {os.path.join(args.out, 'predictions.json')}")


if __name__ == "__main__":
    main()
