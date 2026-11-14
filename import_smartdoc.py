"""Turn the SmartDoc 2015 Challenge 1 dataset into a docclean test set.

    py import_smartdoc.py --smartdoc C:\\path\\to\\smartdoc

Writes, in the docclean folder:
    photos_smartdoc\\              the selected frames, e.g. smartdoc-bg1_letter002_f0071.jpg
    annotations_smartdoc.json     true corners in the same format annotate.py saves
    shot_list_smartdoc.csv        manifest in the same format as shot_list.csv

Then run the usual steps on it:
    py run_pipeline.py --images photos_smartdoc --out results_smartdoc
    py evaluate.py --manifest shot_list_smartdoc.csv --gt annotations_smartdoc.json --pred results_smartdoc\\predictions.json --out report_smartdoc

Frame selection: each clip is about 10 seconds of handheld video, so neighboring
frames are near-duplicates. By default one frame is taken from the middle of each
of the 150 clips (30 documents x 5 backgrounds). --per-video 2 or 3 takes frames
spread evenly across each clip instead. Frames where the page runs outside the image
are skipped in favor of the nearest frame where all four corners are in view.

Dataset: Burie et al., "ICDAR2015 Competition on Smartphone Document Capture and
OCR (SmartDoc)", ICDAR 2015. Licensed CC BY 4.0.
Download frames.tar.gz from https://github.com/jchazalon/smartdoc15-ch1-dataset/releases
"""
import argparse
import csv
import gzip
import json
import os
import shutil
from collections import defaultdict

# Short descriptions of the five SmartDoc backgrounds (edit if you describe them differently)
BACKGROUNDS = {
    "background01": "light wood table",
    "background02": "light gray table",
    "background03": "pink-beige table",
    "background04": "cream table",
    "background05": "cluttered desk",
}
FACTORS = ["perspective", "lighting", "background", "document", "orientation", "quality"]
CORNERS = ["tl", "tr", "br", "bl"]


def load_metadata(root):
    path = os.path.join(root, "metadata.csv.gz")
    if not os.path.exists(path):
        raise SystemExit(f"Could not find {path}. Point --smartdoc at the folder where you "
                         f"extracted frames.tar.gz (it contains metadata.csv.gz and background01..05).")
    clips = defaultdict(list)
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            r["frame_index"] = int(r["frame_index"])
            r["quad"] = [[float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in CORNERS]
            clips[(r["bg_name"], r["model_name"])].append(r)
    for frames in clips.values():
        frames.sort(key=lambda r: r["frame_index"])
    return clips


def inside(quad, w, h, margin):
    return all(margin <= x <= w - 1 - margin and margin <= y <= h - 1 - margin for x, y in quad)


def pick_frames(frames, k, w, h, margin):
    """k evenly spaced targets per clip, each moved to the nearest fully visible frame."""
    n = len(frames)
    targets = [round((i + 1) * n / (k + 1)) for i in range(k)]
    chosen = []
    for t in targets:
        for off in range(n):
            hit = next((j for j in (t - off, t + off)
                        if 0 <= j < n and j not in chosen and inside(frames[j]["quad"], w, h, margin)), None)
            if hit is not None:
                chosen.append(hit)
                break
    return [frames[j] for j in sorted(chosen)]


def orientation(quad):
    """Portrait or landscape as the page appears in the frame: which way its top-to-bottom axis runs."""
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = quad
    dx = (blx + brx - tlx - trx) / 2
    dy = (bly + bry - tly - try_) / 2
    return "portrait" if abs(dy) >= abs(dx) else "landscape"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smartdoc", required=True, help="folder where frames.tar.gz was extracted")
    ap.add_argument("--images", default="photos_smartdoc")
    ap.add_argument("--gt", default="annotations_smartdoc.json")
    ap.add_argument("--manifest", default="shot_list_smartdoc.csv")
    ap.add_argument("--per-video", type=int, default=1, help="frames taken from each clip (default 1)")
    ap.add_argument("--backgrounds", nargs="+", default=list(BACKGROUNDS),
                    help="e.g. --backgrounds background01 background05")
    ap.add_argument("--margin", type=float, default=5, help="px the corners must stay inside the frame")
    args = ap.parse_args()

    clips = load_metadata(args.smartdoc)
    os.makedirs(args.images, exist_ok=True)
    ann, rows, skipped = {}, [], []

    for (bg, model), frames in sorted(clips.items()):
        if bg not in args.backgrounds:
            continue
        w, h = 1920, 1080                                  # all SmartDoc frames are Full HD
        picked = pick_frames(frames, args.per_video, w, h, args.margin)
        if len(picked) < args.per_video:
            skipped.append(f"{bg}/{model}")
        for r in picked:
            bg_num = bg[-1]
            name = f"smartdoc-bg{bg_num}_{model}_f{r['frame_index']:04d}.jpg"
            shutil.copyfile(os.path.join(args.smartdoc, r["image_path"]), os.path.join(args.images, name))
            ann[name] = {"width": w, "height": h, "corners": [[round(x, 1), round(y, 1)] for x, y in r["quad"]]}
            rows.append({"filename": name, "condition": f"smartdoc-bg{bg_num}",
                         "perspective": "varies (handheld)", "lighting": "varies (handheld)",
                         "background": BACKGROUNDS.get(bg, bg), "document": r["modeltype_name"],
                         "orientation": orientation(r["quad"]), "quality": "varies (handheld)",
                         "notes": f"SmartDoc 2015 Ch1 {r['image_path']}"})

    with open(args.gt, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=1)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, ["filename", "condition"] + FACTORS + ["notes"])
        wr.writeheader()
        wr.writerows(rows)

    per_cond = defaultdict(int)
    for r in rows:
        per_cond[r["condition"]] += 1
    print(f"Copied {len(rows)} frames to {os.path.join(args.images, '')}")
    for c, n in sorted(per_cond.items()):
        print(f"  {c}: {n} images")
    if skipped:
        print(f"Fewer frames than requested (page left the frame too often): {', '.join(skipped)}")
    print(f"Ground truth -> {args.gt}\nManifest     -> {args.manifest}")


if __name__ == "__main__":
    main()