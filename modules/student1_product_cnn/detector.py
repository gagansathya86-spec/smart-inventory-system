import os
import json
import time
import numpy as np
import cv2
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "categories.json")
MODEL_ENHANCED = os.path.join(CURRENT_DIR, "yolov8s_17categories_enhanced.pt")
MODEL_BASE = os.path.join(CURRENT_DIR, "yolov8s_17classes.pt")

class FastProductDetector:
    def __init__(self, model_path=None, conf_thresh=0.20, iou_thresh=0.50, imgsz=384):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        
        # Load category metadata
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.categories = cfg["categories"]
        self.category_colors = {
            cat: tuple(color) for cat, color in cfg["category_colors"].items()
        }
        self.prompt_to_category = cfg.get("prompt_to_category", {})

        # Select model
        if model_path is None:
            if os.path.exists(MODEL_ENHANCED):
                model_path = MODEL_ENHANCED
            elif os.path.exists(MODEL_BASE):
                model_path = MODEL_BASE
            else:
                model_path = "yolov8s-worldv2.pt"

        print(f"[Detector] Loading model from: {model_path}...")
        t0 = time.time()
        self.model = YOLO(model_path)
        
        # If model doesn't have custom classes baked in, set them now
        if len(self.model.names) != len(self.prompt_to_category) and hasattr(self.model, "set_classes"):
            print(f"[Detector] Configuring {len(self.prompt_to_category)} open-vocabulary prompts on YOLO-World...")
            self.model.set_classes(list(self.prompt_to_category.keys()))

        print(f"[Detector] Model loaded in {time.time() - t0:.2f}s with {len(self.model.names)} classes.")

        # Performance metrics
        self.last_inference_time_ms = 0.0
        self.fps = 0.0

    def set_confidence(self, conf: float):
        self.conf_thresh = max(0.05, min(0.95, conf))

    def set_imgsz(self, sz: int):
        self.imgsz = sz

    def predict(self, frame: np.ndarray):
        """
        Runs fast inference on a BGR frame and returns:
        - detections: list of dicts with category, conf, box, prompt
        - inference_time_ms: float
        """
        if frame is None or frame.size == 0:
            return [], 0.0

        t0 = time.perf_counter()

        # Run YOLO-World prediction
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device="cpu",
            verbose=False
        )[0]

        inference_time_ms = (time.perf_counter() - t0) * 1000
        self.last_inference_time_ms = inference_time_ms

        raw_detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                prompt_name = self.model.names.get(cls_id, "unknown")
                canonical_cat = self.prompt_to_category.get(prompt_name, prompt_name)
                
                # Check if it maps to one of our 17 categories
                if canonical_cat not in self.categories:
                    # Fallback fuzzy match
                    for c in self.categories:
                        if c in prompt_name.lower():
                            canonical_cat = c
                            break

                color = self.category_colors.get(canonical_cat, (0, 255, 0))

                raw_detections.append({
                    "category": canonical_cat,
                    "prompt": prompt_name,
                    "conf": conf,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "color": color
                })

        # Deduplicate overlapping boxes for the same or competing category
        detections = self._deduplicate_boxes(raw_detections)
        return detections, inference_time_ms

    def _deduplicate_boxes(self, dets, iou_cutoff=0.45):
        if len(dets) <= 1:
            return dets

        # Sort descending by confidence
        dets.sort(key=lambda d: d["conf"], reverse=True)
        keep = []

        for det in dets:
            overlap = False
            b1 = det["box"]
            area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
            if area1 <= 0:
                continue

            for kept in keep:
                b2 = kept["box"]
                # Compute IoU
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih

                if inter_area > 0:
                    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
                    iou = inter_area / float(area1 + area2 - inter_area)
                    if iou > iou_cutoff:
                        overlap = True
                        break

            if not overlap:
                keep.append(det)

        return keep

    def draw_hud(self, frame: np.ndarray, detections: list, display_fps: float = 0.0):
        """
        Draws professional futuristic HUD bounding boxes and badges on frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw category detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            color = det["color"]
            cat_name = det["category"].upper()
            conf_str = f"{det['conf']:.0%}"

            # 1. Main bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # 2. Corner brackets for modern aesthetic
            corner_len = min(20, max(8, int(min(x2 - x1, y2 - y1) * 0.2)))
            thick = 3
            # Top-Left
            cv2.line(out, (x1, y1), (x1 + corner_len, y1), color, thick, cv2.LINE_AA)
            cv2.line(out, (x1, y1), (x1, y1 + corner_len), color, thick, cv2.LINE_AA)
            # Top-Right
            cv2.line(out, (x2, y1), (x2 - corner_len, y1), color, thick, cv2.LINE_AA)
            cv2.line(out, (x2, y1), (x2, y1 + corner_len), color, thick, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(out, (x1, y2), (x1 + corner_len, y2), color, thick, cv2.LINE_AA)
            cv2.line(out, (x1, y2), (x1, y2 - corner_len), color, thick, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, thick, cv2.LINE_AA)
            cv2.line(out, (x2, y2), (x2, y2 - corner_len), color, thick, cv2.LINE_AA)

            # 3. Label badge
            label_text = f" {cat_name} {conf_str} "
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thick = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

            badge_h = th + 10
            badge_y1 = max(0, y1 - badge_h)
            badge_y2 = badge_y1 + badge_h
            badge_x2 = min(w, x1 + tw + 6)

            # Dark translucent pill
            sub_img = out[badge_y1:badge_y2, x1:badge_x2]
            dark_rect = np.full_like(sub_img, 20)
            cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 0, sub_img)

            # Border on badge
            cv2.rectangle(out, (x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)

            # Text
            cv2.putText(
                out, label_text,
                (x1 + 3, badge_y2 - 5),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
            )

        # 4. Top Telemetry Bar
        bar_h = 32
        sub_bar = out[0:bar_h, 0:w]
        black_bar = np.zeros_like(sub_bar)
        cv2.addWeighted(sub_bar, 0.3, black_bar, 0.7, 0, sub_bar)

        # Telemetry text
        det_count = len(detections)
        hud_text = (
            f"FPS: {display_fps:4.1f} | Latency: {self.last_inference_time_ms:4.1f}ms | "
            f"Conf: {self.conf_thresh:.2f} | Detected: {det_count}"
        )
        cv2.putText(
            out, hud_text, (12, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 1, cv2.LINE_AA
        )

        return out
