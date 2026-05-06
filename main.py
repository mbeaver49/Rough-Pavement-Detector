# coding: utf-8
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from skimage import data as skdata

from terrain_core import (
    SafetyThresholds,
    BoundingBox,
    extract_roi_features,
    classify_roi,
)
from gui import TerrainAnnotator, draw_boxes, fit_to_screen, make_horizontal_rois


def batch_classify(
    image: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    thresholds: SafetyThresholds,
) -> Tuple[np.ndarray, List[BoundingBox]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    boxes = []
    for x1, y1, x2, y2 in rois:
        feats = extract_roi_features(gray, x1, y1, x2, y2)
        is_safe, reasons = classify_roi(feats, thresholds)
        boxes.append(
            BoundingBox(
                x1, y1, x2, y2, features=feats, is_safe=is_safe, reasons=reasons
            )
        )

    canvas = image.copy()
    canvas = draw_boxes(canvas, boxes)
    return canvas, boxes


def process_video(video_path: str, out_path: str, args, thresholds: SafetyThresholds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Empty video file.")
        cap.release()
        return

    frame = fit_to_screen(frame, max_w=args.max_w, max_h=args.max_h)
    h, w = frame.shape[:2]

    rois = make_horizontal_rois(
        frame,
        n=args.n_rois,
        height_frac=args.height_frac,
        width_frac=args.width_frac,
        y_pos_frac=args.y_pos,
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing video: {video_path} ({total_frames} frames expected)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    curr_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_frame += 1
        if curr_frame % 30 == 0:
            print(f"  Processed {curr_frame}/{total_frames} frames...")

        frame = fit_to_screen(frame, max_w=args.max_w, max_h=args.max_h)
        annotated_frame, _ = batch_classify(frame, rois, thresholds)
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"\nSaved annotated video to: {out_path}")


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(
        description="Terrain safety annotator via GLCM texture"
    )
    parser.add_argument("--image", default=None, help="Path to a single input image.")
    parser.add_argument("--folder", default=None, help="Path to a folder of images.")
    parser.add_argument("--video", default=None, help="Path to an input video file.")
    parser.add_argument(
        "--output-dir",
        default="terrain_output",
        help="Folder to save annotated images/videos.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive tuning mode (single image only).",
    )
    parser.add_argument(
        "--n-rois", type=int, default=4, help="Number of horizontal ROI boxes."
    )
    parser.add_argument(
        "--height-frac",
        type=float,
        default=0.22,
        help="Box height as fraction of image height.",
    )
    parser.add_argument(
        "--width-frac",
        type=float,
        default=0.75,
        help="Total row width as fraction of image width.",
    )
    parser.add_argument(
        "--y-pos",
        type=float,
        default=0.65,
        help="Vertical centre of ROI row as fraction of image height.",
    )
    parser.add_argument(
        "--max-w", type=int, default=1280, help="Max display width in pixels."
    )
    parser.add_argument(
        "--max-h", type=int, default=720, help="Max display height in pixels."
    )
    parser.add_argument("--max-contrast", type=float, default=50.0)
    parser.add_argument("--max-entropy", type=float, default=3.5)
    parser.add_argument("--max-std-contrast", type=float, default=30.0)
    parser.add_argument("--min-homogeneity", type=float, default=0.0)
    parser.add_argument("--min-correlation", type=float, default=0.0)
    parser.add_argument("--min-energy", type=float, default=0.0)
    args = parser.parse_args()

    thresholds = SafetyThresholds(
        max_contrast=args.max_contrast,
        max_entropy=args.max_entropy,
        max_std_contrast=args.max_std_contrast,
        min_homogeneity=args.min_homogeneity,
        min_correlation=args.min_correlation,
        min_energy=args.min_energy,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(args.video)
        out_name = video_path.stem + "_annotated.mp4"
        save_path = out_dir / out_name
        process_video(str(video_path), str(save_path), args, thresholds)
        return

    image_paths = []
    if args.folder:
        image_paths = sorted(
            [
                p
                for p in Path(args.folder).iterdir()
                if p.suffix.lower() in VALID_EXTENSIONS
            ]
        )
        if not image_paths:
            raise FileNotFoundError(f"No valid images found in: {args.folder}")
    elif args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = [None]

    for idx, path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}] {path or 'demo image'}")

        if path is None:
            img_bgr = cv2.cvtColor(skdata.camera(), cv2.COLOR_GRAY2BGR)
            out_name = "demo_annotated.png"
        else:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                print(f"  ⚠ Could not read, skipping.")
                continue
            out_name = path.stem + "_annotated" + path.suffix

        img_bgr = fit_to_screen(img_bgr, max_w=args.max_w, max_h=args.max_h)

        rois = make_horizontal_rois(
            img_bgr,
            n=args.n_rois,
            height_frac=args.height_frac,
            width_frac=args.width_frac,
            y_pos_frac=args.y_pos,
        )

        if args.interactive and len(image_paths) == 1:
            annotator = TerrainAnnotator(img_bgr, thresholds, initial_rois=rois)
            final_boxes = annotator.run()
            if final_boxes:
                features_list = [b.features for b in final_boxes if b.features is not None]
                if features_list:
                    print("\nROI Extracted Feature Extremes:")
                    print(f"  Max Contrast:     {max(f.mean_contrast for f in features_list):.2f}")
                    print(f"  Max Entropy:      {max(f.mean_entropy for f in features_list):.2f}")
                    print(f"  Max Std Contrast: {max(f.std_contrast for f in features_list):.2f}")
                    print(f"  Min Homogeneity:  {min(f.mean_homogeneity for f in features_list):.2f}")
                    print(f"  Min Correlation:  {min(f.mean_correlation for f in features_list):.2f}")
                    print(f"  Min Energy:       {min(f.mean_energy for f in features_list):.2f}")
        else:
            annotated, _ = batch_classify(img_bgr, rois, thresholds)
            save_path = out_dir / out_name
            cv2.imwrite(str(save_path), annotated)
            print(f"  Saved → {save_path}")

    print(f"\nDone. {len(image_paths)} image(s) processed. Results in: {out_dir}/")


if __name__ == "__main__":
    main()
