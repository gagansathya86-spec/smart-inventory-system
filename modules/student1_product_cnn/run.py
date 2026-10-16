import argparse
import sys
import os
import webbrowser
import threading
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from web_server import start_server
from app_window import run_window_app

def open_browser(url, delay=1.5):
    time.sleep(delay)
    print(f"[Launcher] Opening browser at: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[Launcher] Could not open browser automatically: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fast 17-Category Live Product Detection System")
    parser.add_argument(
        "--mode",
        choices=["web", "window", "both"],
        default="web",
        help="Display mode: 'web' for browser dashboard (default), 'window' for native OpenCV desktop window, 'both' for both."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold (default: 0.20)")
    parser.add_argument("--imgsz", type=int, default=384, help="Inference image resolution (default: 384)")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser in web mode")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("      FAST LIVE PRODUCT RECOGNITION SYSTEM (17 CATEGORIES)")
    print("=" * 70)
    print(f" Mode:            {args.mode.upper()}")
    print(f" Camera Index:    {args.camera}")
    print(f" Confidence:      {args.conf:.2f}")
    print(f" Resolution:      {args.imgsz}px")
    print(f" Target Classes:  17 categories (biscuit, chips, pen, scale, phone,")
    print(f"                  calculator, headphone, laptop, books, pencils,")
    print(f"                  erasers, keyboard, mouse, toothpaste, ball,")
    print(f"                  toothbrush, wires)")
    print("=" * 70 + "\n")

    if args.mode == "window":
        run_window_app(camera_src=args.camera, conf=args.conf, imgsz=args.imgsz)
    elif args.mode == "web":
        if not args.no_browser:
            url = f"http://localhost:{args.port}"
            threading.Thread(target=open_browser, args=(url,), daemon=True).start()
        start_server(host=args.host, port=args.port, camera_src=args.camera, conf=args.conf, imgsz=args.imgsz)
    elif args.mode == "both":
        if not args.no_browser:
            url = f"http://localhost:{args.port}"
            threading.Thread(target=open_browser, args=(url,), daemon=True).start()
        
        # Start web server in background thread
        web_thread = threading.Thread(
            target=start_server,
            kwargs={
                "host": args.host,
                "port": args.port,
                "camera_src": args.camera,
                "conf": args.conf,
                "imgsz": args.imgsz
            },
            daemon=True
        )
        web_thread.start()
        
        # Run desktop window in main thread
        run_window_app(camera_src=args.camera, conf=args.conf, imgsz=args.imgsz)

if __name__ == "__main__":
    main()
