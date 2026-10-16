import os
import glob
import time
import cv2
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
from pipeline import ProductDetectionPipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS_DIR = os.path.join(CURRENT_DIR, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global pipeline instance
pipeline = None

def init_pipeline(camera_src=0, conf=0.20, imgsz=384):
    global pipeline
    if pipeline is None:
        pipeline = ProductDetectionPipeline(camera_src=camera_src, conf=conf, imgsz=imgsz)
        pipeline.start()
    return pipeline

def gen_frames():
    global pipeline
    while True:
        if pipeline is None:
            time.sleep(0.05)
            continue

        frame = pipeline.get_annotated_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Fast JPEG encoding with 80% quality for optimal bandwidth and speed
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/stats")
def api_stats():
    global pipeline
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503
    return jsonify(pipeline.get_stats())

@app.route("/api/set_conf", methods=["POST"])
def api_set_conf():
    global pipeline
    data = request.get_json(silent=True) or {}
    conf = float(data.get("conf", 0.20))
    pipeline.set_confidence(conf)
    return jsonify({"success": True, "conf": pipeline.detector.conf_thresh})

@app.route("/api/set_imgsz", methods=["POST"])
def api_set_imgsz():
    global pipeline
    data = request.get_json(silent=True) or {}
    sz = int(data.get("imgsz", 384))
    pipeline.set_imgsz(sz)
    return jsonify({"success": True, "imgsz": pipeline.detector.imgsz})

@app.route("/api/toggle_pause", methods=["POST"])
def api_toggle_pause():
    global pipeline
    paused = pipeline.toggle_pause()
    return jsonify({"success": True, "paused": paused})

@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    global pipeline
    filepath = pipeline.save_snapshot()
    if filepath and os.path.exists(filepath):
        filename = os.path.basename(filepath)
        return jsonify({"success": True, "filename": filename, "url": f"/snapshots/{filename}"})
    return jsonify({"success": False, "error": "Failed to capture snapshot"}), 500

@app.route("/snapshots/<filename>")
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)

@app.route("/api/snapshots")
def list_snapshots():
    files = glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg"))
    files.sort(key=os.path.getmtime, reverse=True)
    snapshots = [
        {
            "filename": os.path.basename(f),
            "url": f"/snapshots/{os.path.basename(f)}",
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f)))
        }
        for f in files[:20]
    ]
    return jsonify({"snapshots": snapshots})

def start_server(host="0.0.0.0", port=5000, camera_src=0, conf=0.20, imgsz=384):
    init_pipeline(camera_src=camera_src, conf=conf, imgsz=imgsz)
    print(f"\n[Web Server] Running live detection service at: http://localhost:{port}")
    print(f"[Web Server] Access from any browser on this machine or local network.\n")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    start_server()
