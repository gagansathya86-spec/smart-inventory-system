import os
import time
import threading
from datetime import datetime
import cv2
import numpy as np

from camera import ThreadedCamera
from detector import FastProductDetector

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS_DIR = os.path.join(CURRENT_DIR, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

class ProductDetectionPipeline:
    """
    Decoupled Asynchronous Detection Pipeline:
    - Threaded camera capture at 30 FPS.
    - Threaded detection inference running at maximum CPU capacity.
    - Smooth real-time rendering with zero buffer lag.
    """
    def __init__(self, camera_src=0, conf=0.20, imgsz=384):
        self.camera = ThreadedCamera(src=camera_src)
        self.detector = FastProductDetector(conf_thresh=conf, imgsz=imgsz)

        self.running = False
        self.paused = False
        self.lock = threading.Lock()

        # Shared state
        self.current_detections = []
        self.last_inference_time_ms = 0.0
        self.inference_count = 0
        self.display_fps = 0.0
        self.inference_fps = 0.0

        # Stats tracking for the 17 categories
        self.category_counts = {cat: 0 for cat in self.detector.categories}
        self.recent_detections = []  # list of {category, time, conf}
        self.total_detections_history = 0

        # Frame timing
        self._last_display_time = time.time()
        self._display_fps_accum = 0.0
        self._display_frames = 0

    def start(self):
        self.camera.start()
        self.running = True
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()
        return self

    def _inference_loop(self):
        last_t = time.perf_counter()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Run YOLO-World inference
            dets, t_ms = self.detector.predict(frame)

            now_t = time.perf_counter()
            dt = now_t - last_t
            last_t = now_t
            inf_fps = 1.0 / dt if dt > 0 else 0.0

            # Update shared state
            with self.lock:
                self.current_detections = dets
                self.last_inference_time_ms = t_ms
                self.inference_fps = 0.8 * self.inference_fps + 0.2 * inf_fps
                self.inference_count += 1

                # Update category statistics
                now_str = datetime.now().strftime("%H:%M:%S")
                for d in dets:
                    cat = d["category"]
                    self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
                    self.total_detections_history += 1
                    self.recent_detections.append({
                        "category": cat,
                        "time": now_str,
                        "conf": f"{d['conf']:.0%}"
                    })
                # Keep last 50 events
                if len(self.recent_detections) > 50:
                    self.recent_detections = self.recent_detections[-50:]

            # Small yield to prevent thread starvation
            time.sleep(0.002)

    def get_annotated_frame(self):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None

        # Calculate display FPS
        now = time.time()
        dt = now - self._last_display_time
        self._last_display_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self.display_fps = 0.9 * self.display_fps + 0.1 * current_fps

        with self.lock:
            dets = list(self.current_detections)

        # Draw futuristic HUD
        annotated = self.detector.draw_hud(frame, dets, display_fps=self.display_fps)
        return annotated

    def save_snapshot(self):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None

        with self.lock:
            dets = list(self.current_detections)

        annotated = self.detector.draw_hud(frame, dets, display_fps=self.display_fps)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(SNAPSHOTS_DIR, filename)
        cv2.imwrite(filepath, annotated)
        print(f"[Snapshot] Saved to: {filepath}")
        return filepath

    def get_stats(self):
        with self.lock:
            # Active categories right now
            active_cats = list(set(d["category"] for d in self.current_detections))
            return {
                "display_fps": round(self.display_fps, 1),
                "inference_fps": round(self.inference_fps, 1),
                "inference_time_ms": round(self.last_inference_time_ms, 1),
                "active_count": len(self.current_detections),
                "active_categories": active_cats,
                "current_detections": [
                    {
                        "category": d["category"],
                        "conf": round(d["conf"], 2),
                        "box": d["box"]
                    }
                    for d in self.current_detections
                ],
                "all_categories": self.detector.categories,
                "category_counts": self.category_counts,
                "recent_history": self.recent_detections[-12:][::-1],
                "paused": self.paused,
                "conf_thresh": self.detector.conf_thresh,
                "imgsz": self.detector.imgsz
            }

    def set_confidence(self, conf: float):
        self.detector.set_confidence(conf)

    def set_imgsz(self, sz: int):
        self.detector.set_imgsz(sz)

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def stop(self):
        self.running = False
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.camera.stop()
