"""Score automatic detections against hand-labeled corners.

    python evaluate.py --manifest shot_list.csv --gt annotations.json \
                       --pred results/predictions.json --out report

Metrics for each image (all three requested in the proposal feedback):
  corner_err_pct   mean distance between detected and true corners, as % of image diagonal
  max_corner_pct   worst of the four corners, as % of image diagonal
  iou              intersection over union of detected and true page regions
  ok_iou           iou >= --iou-thr (default 0.90)       <- the definition of "correctly located"
  ok_corners       all four corners within --tol % of the diagonal (default 2%)

A missed detection counts as a failure with IoU 0. Corner error is undefined for a miss
and is left out of the averages, which is why the miss rate is reported next to it.

Outputs: per_image.csv, by_condition.csv, by_factor.csv, report.md
"""
import argparse
import csv
import math
import os

import cv2
import numpy as np

from common import FACTORS, load_json, load_manifest, order_corners


def corner_errors(gt, pred):
    """Per-corner distances, matched over the 4 cyclic rotations so a labeling
    order mistake can never inflate the error."""
    gt, pred = order_corners(gt), order_corners(pred)
    best = None
    for k in range(4):
        d = np.linalg.norm(gt - np.roll(pred, k, axis=0), axis=1)
        if best is None or d.mean() < best.mean():
            best = d
    return best


def quad_iou(a, b):
    a = cv2.convexHull(order_corners(a)).astype(np.float32)
    b = cv2.convexHull(order_corners(b)).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def summarize(rows):
    n = len(rows)
    hits = [r for r in rows if r["detected"]]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {
        "n": n,
        "detected_pct": 100.0 * len(hits) / n if n else float("nan"),
        "ok_iou_pct": 100.0 * sum(r["ok_iou"] for r in rows) / n if n else float("nan"),
        "ok_corners_pct": 100.0 * sum(r["ok_corners"] for r in rows) / n if n else float("nan"),
        "mean_iou": mean([r["iou"] for r in rows]),
        "mean_corner_err_pct": mean([r["corner_err_pct"] for r in hits]),
    }


def fmt(v, d=1):
    return "n/a" if isinstance(v, float) and math.isnan(v) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k], 3) if isinstance(r.get(k), float) else r.get(k) for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="report")
    ap.add_argument("--iou-thr", type=float, default=0.90)
    ap.add_argument("--tol", type=float, default=2.0, help="corner tolerance, %% of image diagonal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt_all, pred_all = load_json(args.gt), load_json(args.pred)
    rows, missing = [], []
    for m in load_manifest(args.manifest):
        name = m["filename"]
        g = gt_all.get(name)
        if not g or not g.get("corners"):
            missing.append(name)
            continue
        diag = math.hypot(g["width"], g["height"])
        p = (pred_all.get(name) or {}).get("corners")
        r = {**m, "detected": p is not None, "method": (pred_all.get(name) or {}).get("method") or "none"}
        if p is None:
            r.update(iou=0.0, corner_err_pct=float("nan"), max_corner_pct=float("nan"))
        else:
            d = corner_errors(g["corners"], p) / diag * 100.0
            r.update(iou=quad_iou(g["corners"], p), corner_err_pct=float(d.mean()), max_corner_pct=float(d.max()))
        r["ok_iou"] = r["iou"] >= args.iou_thr
        r["ok_corners"] = r["detected"] and r["max_corner_pct"] <= args.tol
        rows.append(r)

    if not rows:
        raise SystemExit("No images had both a manifest row and ground-truth corners.")

    write_csv(os.path.join(args.out, "per_image.csv"), rows,
              ["filename", "condition"] + FACTORS +
              ["detected", "method", "iou", "corner_err_pct", "max_corner_pct", "ok_iou", "ok_corners"])

    stat_cols = ["n", "detected_pct", "ok_iou_pct", "ok_corners_pct", "mean_iou", "mean_corner_err_pct"]

    # One row per shooting condition (baseline + each one-factor change)
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    by_cond = [{"condition": c, **summarize([r for r in rows if r["condition"] == c])} for c in conds]
    write_csv(os.path.join(args.out, "by_condition.csv"), by_cond, ["condition"] + stat_cols)

    # One row per factor level across every image (pools the baseline into each factor's default level)
    by_factor = []
    for f in FACTORS:
        for lvl in dict.fromkeys(r[f] for r in rows):
            by_factor.append({"factor": f, "level": lvl, **summarize([r for r in rows if r[f] == lvl])})
    write_csv(os.path.join(args.out, "by_factor.csv"), by_factor, ["factor", "level"] + stat_cols)

    overall = summarize(rows)
    base = next((c for c in by_cond if c["condition"] == "baseline"), None)
    worst = min((c for c in by_cond if c["condition"] != "baseline"),
                key=lambda c: (c["ok_iou_pct"], -c["mean_corner_err_pct"] if not math.isnan(c["mean_corner_err_pct"]) else 0),
                default=None)

    lines = ["# Document Detection Evaluation", "",
             f"A detection counts as **correct** when IoU with the hand-labeled page is at least "
             f"{args.iou_thr:.2f}. Corner accuracy is the share of images where all four corners fall "
             f"within {args.tol:g}% of the image diagonal.", "",
             f"The system correctly located the document in **{fmt(overall['ok_iou_pct'])}%** of "
             f"{overall['n']} images (mean IoU {fmt(overall['mean_iou'], 3)}, mean corner error "
             f"{fmt(overall['mean_corner_err_pct'], 2)}% of the diagonal)."]
    if base and worst:
        lines.append(f"Accuracy was {fmt(base['ok_iou_pct'])}% at baseline and lowest under "
                     f"**{worst['condition']}** ({fmt(worst['ok_iou_pct'])}%).")
    lines += ["", "## By condition", "",
              "| Condition | n | Detected % | Correct (IoU) % | Corners OK % | Mean IoU | Mean corner err % |",
              "|---|---|---|---|---|---|---|"]
    for c in by_cond:
        lines.append(f"| {c['condition']} | {c['n']} | {fmt(c['detected_pct'])} | {fmt(c['ok_iou_pct'])} | "
                     f"{fmt(c['ok_corners_pct'])} | {fmt(c['mean_iou'], 3)} | {fmt(c['mean_corner_err_pct'], 2)} |")
    fails = [r for r in rows if not r["ok_iou"]]
    if fails:
        lines += ["", "## Failures", ""]
        lines += [f"- {r['filename']} ({r['condition']}): " +
                  ("no page found" if not r["detected"] else f"IoU {r['iou']:.3f}") for r in fails]
    if missing:
        lines += ["", f"_{len(missing)} manifest images had no ground truth and were left out: "
                      f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}_"]
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nCSV files and report.md written to {args.out}/")


if __name__ == "__main__":
    main()
