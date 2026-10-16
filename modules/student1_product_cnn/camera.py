import cv2
import threading
import time
import numpy as np

class ThreadedCamera:
    """
    High-performance threaded camera stream that avoids OpenCV buffer lag
    by continuously dropping stale frames and only caching the latest one.
    """
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_frame_time = 0.0
        self.is_synthetic = False

        self._init_camera()

    def _init_camera(self):
        # Try DirectShow on Windows for fastest hardware init
        print(f"[Camera] Initializing camera source {self.src} with DirectShow...")
        try:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            print(f"[Camera] DirectShow failed, trying default backend for {self.src}...")
            self.cap = cv2.VideoCapture(self.src)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = frame
                self.is_synthetic = False
                print(f"[Camera] Connected successfully! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return
            else:
                print("[Camera] Warning: Camera opened but failed to read initial frame.")

        print(f"[Camera] Camera {self.src} not accessible. Using virtual test pattern generator.")
        self.is_synthetic = True
        self.latest_frame = self._generate_test_frame()

    def _generate_test_frame(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (30, 25, 20)
        cv2.putText(
            img, "TEST FEED (NO WEBCAM)", (50, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2
        )
        return img

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        while self.running:
            if not self.is_synthetic and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.last_frame_time = time.time()
                else:
                    time.sleep(0.01)
            else:
                # Synthetic animation
                with self.lock:
                    self.latest_frame = self._generate_test_frame()
                    self.last_frame_time = time.time()
                time.sleep(0.033)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[Camera] Released camera hardware.")
