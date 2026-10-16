import time
import cv2
import numpy as np
from pipeline import ProductDetectionPipeline

def run_window_app(camera_src=0, conf=0.20, imgsz=384):
    print("=" * 65)
    print("  FAST LIVE PRODUCT DETECTION (17 CATEGORIES) - NATIVE WINDOW")
    print("=" * 65)
    print("Controls:")
    print("  [Q] or [ESC] : Quit")
    print("  [S]          : Save Snapshot to 'snapshots/' folder")
    print("  [+] / [-]    : Increase / Decrease Confidence Threshold")
    print("  [SPACE] / [P]: Pause / Resume Detection")
    print("  [C]          : Toggle Category Statistics Sidebar")
    print("=" * 65)

    pipeline = ProductDetectionPipeline(camera_src=camera_src, conf=conf, imgsz=imgsz)
    pipeline.start()

    window_name = "Fast 17-Category Product Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    show_sidebar = True

    try:
        while True:
            frame = pipeline.get_annotated_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Optionally draw stats sidebar
            if show_sidebar:
                stats = pipeline.get_stats()
                sidebar_w = 220
                canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
                canvas[:, :w] = frame

                # Dark sidebar
                canvas[:, w:] = (22, 17, 14)
                cv2.line(canvas, (w, 0), (w, h), (50, 40, 35), 1)

                # Sidebar Header
                cv2.putText(canvas, "17 CATEGORIES", (w + 14, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2, cv2.LINE_AA)

                # Categories list with counts
                cats = stats["all_categories"]
                active_set = set(stats["active_categories"])
                counts = stats["category_counts"]

                y_start = 55
                row_h = (h - 90) // len(cats)
                for idx, cat in enumerate(cats):
                    y = y_start + idx * row_h
                    is_active = cat in active_set
                    dot_color = (0, 255, 128) if is_active else (80, 80, 80)
                    text_color = (255, 255, 255) if is_active else (160, 160, 160)

                    # Status indicator dot
                    cv2.circle(canvas, (w + 18, y - 4), 4, dot_color, -1, cv2.LINE_AA)

                    # Category name and count
                    cat_display = f"{cat[:11]:11s} : {counts.get(cat, 0)}"
                    cv2.putText(canvas, cat_display, (w + 30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color, 1, cv2.LINE_AA)

                # Bottom Controls Reminder
                cv2.putText(canvas, "[S] Snap  [C] Side  [Q] Exit", (w + 12, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

                display_img = canvas
            else:
                display_img = frame

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):
                path = pipeline.save_snapshot()
                print(f"[Snapshot] Saved to: {path}")
            elif key == ord('+') or key == ord('='):
                new_conf = min(0.90, pipeline.detector.conf_thresh + 0.05)
                pipeline.set_confidence(new_conf)
                print(f"[Controls] Confidence threshold set to: {new_conf:.2f}")
            elif key == ord('-') or key == ord('_'):
                new_conf = max(0.05, pipeline.detector.conf_thresh - 0.05)
                pipeline.set_confidence(new_conf)
                print(f"[Controls] Confidence threshold set to: {new_conf:.2f}")
            elif key == ord('c'):
                show_sidebar = not show_sidebar
            elif key == ord('p') or key == 32:  # 'p' or SPACE
                paused = pipeline.toggle_pause()
                print(f"[Controls] {'PAUSED' if paused else 'RESUMED'}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Window App] Shutdown complete.")

if __name__ == "__main__":
    run_window_app()
