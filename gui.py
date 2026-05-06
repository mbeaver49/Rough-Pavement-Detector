import cv2
import numpy as np
import tkinter as tk
from typing import List, Tuple, Optional
from terrain_core import (
    BoundingBox,
    SafetyThresholds,
    extract_roi_features,
    classify_roi,
)


def fit_to_screen(image, max_w=1280, max_h=720):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def make_horizontal_rois(image, n=4, height_frac=0.4, width_frac=0.8, y_pos_frac=0.65):
    h, w = image.shape[:2]
    total_w = int(w * width_frac)
    box_w = total_w // n
    box_h = int(h * height_frac)
    x_start = (w - total_w) // 2
    y_centre = int(h * y_pos_frac)
    y_start = y_centre - box_h // 2
    y_start = max(0, min(y_start, h - box_h))

    rois = []
    for i in range(n):
        x1 = x_start + i * box_w
        y1 = y_start
        x2 = x1 + box_w
        y2 = y1 + box_h
        rois.append((x1, y1, x2, y2))
    return rois


def draw_boxes(canvas: np.ndarray, boxes: List[BoundingBox]) -> np.ndarray:
    overlay = canvas.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        color = box.color

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        overlay = canvas.copy()

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        badge_text = f"ROI-{idx+1} {box.status_text}"
        (tw, th), baseline = cv2.getTextSize(
            badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        badge_y = max(y1 - 4, th + 4)
        cv2.rectangle(
            canvas, (x1, badge_y - th - 4), (x1 + tw + 6, badge_y + baseline), color, -1
        )
        cv2.putText(
            canvas,
            badge_text,
            (x1 + 3, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if box.features and (x2 - x1) > 80 and (y2 - y1) > 60:
            lines = [
                f"Ctr:{box.features.mean_contrast:.1f}",
                f"Ent:{box.features.mean_entropy:.2f}",
                f"Hom:{box.features.mean_homogeneity:.2f}",
                f"Cor:{box.features.mean_correlation:.2f}",
                f"Eng:{box.features.mean_energy:.2f}",
            ]
            for k, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (x1 + 4, y1 + 16 + k * 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return canvas


def draw_hint_bar(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    hints = "Z: undo  |  S: save  |  Q: quit"
    cv2.rectangle(canvas, (0, h - 22), (w, h), (30, 30, 30), -1)
    cv2.putText(
        canvas,
        hints,
        (6, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


class TerrainAnnotator:
    def __init__(
        self,
        image: np.ndarray,
        thresholds: SafetyThresholds,
        initial_rois: List[Tuple[int, int, int, int]] = None,
    ):
        self.orig = image.copy()
        self.thresholds = thresholds
        self.gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        self.boxes: List[BoundingBox] = []

        if initial_rois:
            for x1, y1, x2, y2 in initial_rois:
                self.boxes.append(BoundingBox(x1, y1, x2, y2))

        self._drawing = False
        self._pt1 = None
        self._pt2 = None
        self._pending = None
        self.window_name = "Terrain Safety Annotator"

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._pt1 = (x, y)
            self._pt2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._pt2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1, y1 = min(self._pt1[0], x), min(self._pt1[1], y)
            x2, y2 = max(self._pt1[0], x), max(self._pt1[1], y)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                self._pending = BoundingBox(x1, y1, x2, y2)
                feats = extract_roi_features(self.gray, x1, y1, x2, y2)
                is_safe, reasons = classify_roi(feats, self.thresholds)
                self._pending.features = feats
                self._pending.is_safe = is_safe
                self._pending.reasons = reasons
            self._pt1 = self._pt2 = None

    def _classify_all(self):
        for box in self.boxes:
            if box.features is None:
                box.features = extract_roi_features(
                    self.gray, box.x1, box.y1, box.x2, box.y2
                )
            is_safe, reasons = classify_roi(box.features, self.thresholds)
            box.is_safe = is_safe
            box.reasons = reasons

    def _apply_thresholds(self):
        try:
            self.thresholds.max_contrast = float(self.entry_max_contrast.get())
            self.thresholds.max_entropy = float(self.entry_max_entropy.get())
            self.thresholds.max_std_contrast = float(self.entry_max_std_contrast.get())
            self.thresholds.min_homogeneity = float(self.entry_min_homogeneity.get())
            self.thresholds.min_correlation = float(self.entry_min_correlation.get())
            self.thresholds.min_energy = float(self.entry_min_energy.get())

            self._classify_all()

            if self._pending and self._pending.features:
                is_safe, reasons = classify_roi(self._pending.features, self.thresholds)
                self._pending.is_safe = is_safe
                self._pending.reasons = reasons
        except ValueError:
            print("Invalid input for thresholds. Please enter valid numbers.")

    def _render(self) -> np.ndarray:
        canvas = self.orig.copy()
        canvas = draw_boxes(canvas, self.boxes)
        if self._drawing and self._pt1 and self._pt2:
            cv2.rectangle(canvas, self._pt1, self._pt2, (200, 200, 0), 1)
        if self._pending:
            color = (
                self._pending.color
                if self._pending.is_safe is not None
                else (200, 200, 0)
            )
            cv2.rectangle(
                canvas,
                (self._pending.x1, self._pending.y1),
                (self._pending.x2, self._pending.y2),
                color,
                2,
            )
            cv2.putText(
                canvas,
                "SPACE: confirm / Z: discard",
                (self._pending.x1, max(self._pending.y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            if self._pending.features:
                f = self._pending.features
                stats = f"Ctr:{f.mean_contrast:.1f} Ent:{f.mean_entropy:.2f} Hom:{f.mean_homogeneity:.2f} Cor:{f.mean_correlation:.2f} Eng:{f.mean_energy:.2f}"
                cv2.putText(
                    canvas,
                    stats,
                    (self._pending.x1, self._pending.y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        canvas = draw_hint_bar(canvas)
        return canvas

    def run(self, window_name: str = "Terrain Safety Annotator") -> List[BoundingBox]:
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        self.root = tk.Tk()
        self.root.title("Threshold Settings")

        tk.Label(self.root, text="Max Contrast:").grid(
            row=0, column=0, padx=5, pady=2, sticky="e"
        )
        self.entry_max_contrast = tk.Entry(self.root)
        self.entry_max_contrast.insert(0, "inf")
        self.entry_max_contrast.grid(row=0, column=1, padx=5, pady=2)

        tk.Label(self.root, text="Max Entropy:").grid(
            row=1, column=0, padx=5, pady=2, sticky="e"
        )
        self.entry_max_entropy = tk.Entry(self.root)
        self.entry_max_entropy.insert(0, "inf")
        self.entry_max_entropy.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(self.root, text="Max Std Contrast:").grid(
            row=2, column=0, padx=5, pady=2, sticky="e"
        )
        self.entry_max_std_contrast = tk.Entry(self.root)
        self.entry_max_std_contrast.insert(0, "inf")
        self.entry_max_std_contrast.grid(row=2, column=1, padx=5, pady=2)

        tk.Label(self.root, text="Min Homogeneity:").grid(
            row=3, column=0, padx=5, pady=2, sticky="e"
        )
        self.entry_min_homogeneity = tk.Entry(self.root)
        self.entry_min_homogeneity.insert(0, "-inf")
        self.entry_min_homogeneity.grid(row=3, column=1, padx=5, pady=2)

        tk.Label(self.root, text="Min Correlation:").grid(
            row=4, column=0, padx=5, pady=2, sticky="e"
        )
        self.entry_min_correlation = tk.Entry(self.root)
        self.entry_min_correlation.insert(0, "-inf")
        self.entry_min_correlation.grid(row=4, column=1, padx=5, pady=2)

        tk.Label(self.root, text="Min Energy:").grid(
            row=5, column=0, padx=5, pady=2, sticky="e"
        )
        self.entry_min_energy = tk.Entry(self.root)
        self.entry_min_energy.insert(0, "-inf")
        self.entry_min_energy.grid(row=5, column=1, padx=5, pady=2)

        apply_btn = tk.Button(self.root, text="Apply", command=self._apply_thresholds)
        apply_btn.grid(row=6, column=0, columnspan=2, pady=10)

        self._apply_thresholds()

        while True:
            try:
                self.root.update()
            except tk.TclError:
                pass  # The Tkinter window might have been closed manually by the user

            cv2.imshow(self.window_name, self._render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord(" "), 13):
                if self._pending:
                    self.boxes.append(self._pending)
                    self._pending = None
            elif key == ord("z"):
                if self._pending:
                    self._pending = None
                elif self.boxes:
                    self.boxes.pop()
            elif key == ord("c"):
                self._classify_all()
            elif key == ord("s"):
                out_path = "terrain_annotated.png"
                cv2.imwrite(out_path, self._render())
                print(
                    f"[Saved] {out_path} with Ctr:{self.thresholds.max_contrast} Ent:{self.thresholds.max_entropy}"
                )
            elif key in (ord("q"), 27):
                break

        try:
            self.root.destroy()
        except tk.TclError:
            pass

        cv2.destroyAllWindows()
        return self.boxes
