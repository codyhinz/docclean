"""Hand-label the four true page corners in each photo.

    python annotate.py --images photos --out annotations.json

Click the four corners in any order (they are sorted clockwise automatically).
A magnifier in the top-left corner shows the area under the cursor for precise clicks.

    Enter / Space   save and go to the next image (needs 4 points)
    u               undo last point
    r               reset points on this image
    s               skip this image (no page visible / unusable)
    b               go back one image
    q / Esc         save and quit

Progress is saved after every image, so you can quit and resume any time.
Already-labeled images are skipped unless you pass --redo.
"""
import argparse
import os

import cv2
import numpy as np

from common import list_images, load_json, order_corners, save_json

WIN = "annotate"
MAX_W, MAX_H = 1400, 900
ZOOM, LOUPE = 4, 160


def fit_scale(w, h):
    return min(1.0, MAX_W / w, MAX_H / h)


def draw(base, pts, cursor, scale, full):
    view = base.copy()
    for i, (x, y) in enumerate(pts):
        p = (int(x * scale), int(y * scale))
        cv2.circle(view, p, 6, (0, 0, 255), 2)
        cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if len(pts) == 4:
        poly = (order_corners(pts) * scale).astype(np.int32)
        cv2.polylines(view, [poly], True, (0, 255, 0), 2)
    if cursor is not None:                                      # magnifier from full-res image
        cx, cy = int(cursor[0] / scale), int(cursor[1] / scale)
        r = LOUPE // (2 * ZOOM)
        H, W = full.shape[:2]
        crop = full[max(cy - r, 0):min(cy + r, H), max(cx - r, 0):min(cx + r, W)]
        if crop.size:
            crop = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            ch, cw = crop.shape[:2]
            view[0:ch, 0:cw] = crop
            cv2.line(view, (cw // 2, 0), (cw // 2, ch), (0, 255, 255), 1)
            cv2.line(view, (0, ch // 2), (cw, ch // 2), (0, 255, 255), 1)
    msg = f"{len(pts)}/4 points   Enter=save  u=undo  r=reset  s=skip  b=back  q=quit"
    cv2.putText(view, msg, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(view, msg, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="annotations.json")
    ap.add_argument("--redo", action="store_true", help="re-label images that already have corners")
    args = ap.parse_args()

    ann = load_json(args.out)
    names = list_images(args.images)
    todo = names if args.redo else [n for n in names if n not in ann]
    print(f"{len(names)} images found, {len(todo)} left to label.")
    if not todo:
        return

    state = {"pts": [], "cursor": None}

    def on_mouse(event, x, y, flags, param):
        state["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(state["pts"]) < 4:
            state["pts"].append((x / state["scale"], y / state["scale"]))

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    i = 0
    while 0 <= i < len(todo):
        name = todo[i]
        full = cv2.imread(os.path.join(args.images, name))
        if full is None:
            print(f"  could not read {name}")
            i += 1
            continue
        h, w = full.shape[:2]
        state["scale"] = fit_scale(w, h)
        base = cv2.resize(full, None, fx=state["scale"], fy=state["scale"], interpolation=cv2.INTER_AREA)
        state["pts"] = [tuple(p) for p in ann.get(name, {}).get("corners") or []]
        cv2.setWindowTitle(WIN, f"[{i + 1}/{len(todo)}] {name}")

        while True:
            cv2.imshow(WIN, draw(base, state["pts"], state["cursor"], state["scale"], full))
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32) and len(state["pts"]) == 4:
                ann[name] = {"width": w, "height": h,
                             "corners": order_corners(state["pts"]).round(1).tolist()}
                save_json(args.out, ann)
                i += 1
                break
            if key == ord("u") and state["pts"]:
                state["pts"].pop()
            elif key == ord("r"):
                state["pts"] = []
            elif key == ord("s"):
                ann[name] = {"width": w, "height": h, "corners": None, "skipped": True}
                save_json(args.out, ann)
                i += 1
                break
            elif key == ord("b"):
                i = max(0, i - 1)
                break
            elif key in (ord("q"), 27):
                save_json(args.out, ann)
                cv2.destroyAllWindows()
                print(f"Saved {len(ann)} annotations to {args.out}")
                return
    cv2.destroyAllWindows()
    print(f"Done. Saved {len(ann)} annotations to {args.out}")


if __name__ == "__main__":
    main()
