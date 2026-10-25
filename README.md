# Document Photo Cleanup

Classical pipeline that finds a paper document in a phone photo, flattens it, and evens out the lighting, plus the evaluation setup requested in the proposal feedback.

## Setup

```
pip install -r requirements.txt
or
py -m pip install -r requirements.txt

```

iPhone users: photos must be JPEG, not HEIC. Set Settings > Camera > Formats > Most Compatible before shooting.

## Workflow

1. Shoot the 60 photos in `shot_list.csv` and rename each to its `filename` (e.g. `lighting-shadow_03.jpg`). Put them in a folder called `photos/`.
2. Label the true corners: `python annotate.py --images photos --out annotations.json`
3. Run the pipeline: `python run_pipeline.py --images photos --out results`
4. Score it: `python evaluate.py --manifest shot_list.csv --gt annotations.json --pred results/predictions.json --out report`

`report/report.md` gives the overall and per-condition results. `by_condition.csv`, `by_factor.csv`, and `per_image.csv` hold the full numbers.

## Dataset design

Each of the six factors from the feedback has a default level. A **baseline** set uses every default. Each other set changes exactly one factor and holds the rest at baseline, so any drop in accuracy can be traced to that one factor. This gives 12 conditions with 5 photos each (60 total), instead of the 486 needed to cover every combination.

| Factor | Baseline | Variations |
|---|---|---|
| Perspective | small (under 15° from straight down) | medium (15–30°), large (30–45°) |
| Lighting | uniform (overhead room light, no shadow) | shadow (your hand or phone shadow across part of the page), uneven (single lamp off to one side) |
| Background | dark (dark table, towel, or cloth) | light (white counter or bedsheet), textured (wood grain, patterned fabric) |
| Document | printed page | handwritten page, receipt |
| Orientation | portrait | landscape |
| Quality | good (steady, in focus) | moderate (slight shake or dim room), poor (visible blur or grain) |

Tips: use 5 different documents within each condition so results do not hinge on one page. Keep the whole page inside the frame. Use the `notes` column for anything unusual.

## What "correctly located" means

Four true corners are hand-labeled per image. For each detection the evaluator computes:

- **Corner-point error:** mean distance between detected and true corners, as a percent of the image diagonal (so phone resolution does not matter).
- **IoU:** overlap between the detected and true page regions.
- **Corner accuracy:** whether all four corners fall within 2% of the diagonal.

A page counts as **correctly located when IoU is at least 0.90**. A missed detection counts as a failure with IoU 0. Thresholds can be changed with `--iou-thr` and `--tol`.

## Pipeline (pipeline.py)

1. **Detection:** Gaussian smoothing, Canny edges with thresholds set from the median intensity, morphological closing, then the largest convex four-sided contour. Falls back to Otsu thresholding if no edge quadrilateral is found.
2. **Perspective correction:** four-point homography to a straight-on page.
3. **Lighting correction:** background estimated with a large morphological closing plus median blur, divided out of the page to remove shadows and gradients, then a percentile contrast stretch. `--mode binary` adds an Otsu threshold for pure black and white.

The manual-corner fallback from the proposal: `python run_pipeline.py --images photos --out results --manual annotations.json` uses hand-labeled corners when automatic detection fails. Those images are still scored as failures.

## Known limitation

In a synthetic test run, detection was correct on 100% of images in every condition except light backgrounds, where it dropped to 0% (it locked onto the block of text instead of the page edge). This is the low-contrast case the proposal scoped out, and it is the first thing to measure on real photos.
