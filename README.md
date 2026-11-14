# Document Photo Cleanup

A classical image processing pipeline that finds a paper document in a phone photo, flattens it to a straight-on view, and evens out the lighting. The repo also includes the evaluation setup requested in the proposal feedback, run against the public SmartDoc 2015 Challenge 1 dataset.

## Requirements

- Python 3.9 or newer
- About 1 GB of free disk space for the dataset

Install the two dependencies (OpenCV and NumPy):

```
pip install -r requirements.txt
```

On Windows, use `py -m pip install -r requirements.txt` if `pip` is not on your PATH. The same goes for the commands below: `py` works anywhere `python` is shown.

## Quick start (SmartDoc evaluation)

The dataset is not included in the repo because of its size. All four steps run from inside the `docclean` folder.

**1. Download the dataset.** Get `frames.tar.gz` from the SmartDoc 2015 Challenge 1 releases page:
https://github.com/jchazalon/smartdoc15-ch1-dataset/releases

Extract it into a folder named `smartdoc` inside this repo. When you're done, `smartdoc/` should contain `metadata.csv.gz` and the folders `background01` through `background05`.

**2. Pull the test frames out of the dataset.**

```
python import_smartdoc.py --smartdoc smartdoc
```

This copies 150 frames (one from each of the 30 documents on each of the 5 backgrounds) into `photos_smartdoc/`. It also writes the true page corners to `annotations_smartdoc.json` and the image list to `shot_list_smartdoc.csv`. Both of those files are already in the repo, and this step rebuilds them the same way.

**3. Run the pipeline.**

```
python run_pipeline.py --images photos_smartdoc --out results_smartdoc
```

Cleaned pages are saved to `results_smartdoc/cleaned/`, and the detected corners to `results_smartdoc/predictions.json`.

**4. Score the results.**

```
python evaluate.py --manifest shot_list_smartdoc.csv --gt annotations_smartdoc.json --pred results_smartdoc/predictions.json --out report_smartdoc
```

Open `report_smartdoc/report.md` for the summary, with overall results and a breakdown by background. The full numbers are in `per_image.csv`, `by_condition.csv`, and `by_factor.csv` in the same folder.

## Side-by-side demo images

```
python demo_compare.py --images photos_smartdoc --out demo --gt annotations_smartdoc.json
```

Each output image shows the original frame next to the cleaned page. The green outline is what the pipeline detected, and the blue outline is the true page with its IoU score. Add `--show` to step through them in a window (any key moves to the next one), or use `--only <filename> ...` to make just a few.

## What "correctly located" means

Each image has four labeled true corners. For every detection the evaluator computes:

- **Corner-point error:** the mean distance between the detected and true corners, as a percent of the image diagonal, so camera resolution does not matter.
- **IoU:** the overlap between the detected and true page regions.
- **Corner accuracy:** whether all four corners fall within 2% of the diagonal.

A page counts as **correctly located when IoU is at least 0.90**. A missed detection counts as a failure with an IoU of 0. The thresholds can be changed with `--iou-thr` and `--tol` on `evaluate.py`.

## How the pipeline works (pipeline.py)

1. **Detection:** Gaussian smoothing, then Canny edges with thresholds set from the median intensity, then morphological closing. The pipeline keeps the largest convex four-sided contour. If no edge quadrilateral is found, it falls back to Otsu thresholding.
2. **Perspective correction:** a four-point homography maps the page to a straight-on view.
3. **Lighting correction:** the background is estimated with a large morphological closing plus a median blur and divided out of the page to remove shadows and gradients, followed by a percentile contrast stretch. Running with `--mode binary` adds an Otsu threshold for pure black and white output.

**Manual fallback (from the proposal):** `run_pipeline.py --manual annotations_smartdoc.json` uses the hand-labeled corners to produce a cleaned page when automatic detection fails. Those images are still scored as failures, so the evaluation is unaffected.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Detection, perspective correction, and lighting correction |
| `run_pipeline.py` | Runs the pipeline over a folder of images |
| `evaluate.py` | Scores detections against the true corners |
| `demo_compare.py` | Makes side-by-side before and after images |
| `import_smartdoc.py` | Builds the test set from the SmartDoc dataset |
| `annotate.py` | Click-to-label tool for marking true corners on your own photos |
| `common.py` | Shared helpers |
| `shot_list.csv` | Planned 60-photo shot list for a self-collected test set (see below) |

## Optional: testing on your own photos

`shot_list.csv` describes a 60-photo set built to test one factor at a time: perspective, lighting, background, document type, orientation, and image quality. Each factor is varied from a baseline while the rest stay fixed. To use it, put the photos in `photos/` named as in the CSV, then label the corners and run the same steps:

```
python annotate.py --images photos --out annotations.json
python run_pipeline.py --images photos --out results
python evaluate.py --manifest shot_list.csv --gt annotations.json --pred results/predictions.json --out report
```

In `annotate.py`, click the four corners in any order. Enter saves and moves on, `u` undoes, `s` skips, and `q` quits. Progress is saved after every image. iPhone photos must be JPEG, not HEIC (Settings > Camera > Formats > Most Compatible).

## Dataset credit

Burie et al., "ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)," ICDAR 2015. Licensed CC BY 4.0.