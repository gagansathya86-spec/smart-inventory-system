# Student 1 Module: Fast Live Product Recognition (17 Categories)

A high-performance real-time computer vision subsystem for live product recognition across 17 distinct categories with ultra-low latency.

---

## 🌟 Key Features

- **Blazing Fast Live Detection**: 18–30+ FPS real-time webcam inference on CPU/GPU via open-vocabulary YOLO-World v2 with pre-baked prompt embeddings.
- **Zero Gradio Bottleneck**:
  - **Live Web Dashboard** with low-latency multipart MJPEG streaming at `http://localhost:5000`.
  - **Native DirectX / OpenCV Desktop GUI** with zero-network display and interactive keyboard shortcuts.
- **17 Product Categories Detected**:
  1. `biscuit` (Biscuits, Oreo, Cookies)
  2. `chips` (Potato chips, Lay's, Snack packets)
  3. `pen` (Ballpoint pens, Markers, Writing pens)
  4. `scale` (Rulers, Measuring scales)
  5. `phone` (Smartphones, Mobile phones)
  6. `calculator` (Handheld & scientific calculators)
  7. `headphone` (Headsets, Over-ear headphones, Earphones)
  8. `laptop` (Laptops, Notebooks)
  9. `books` (Books, Notebooks, Textbooks)
  10. `pencils` (Pencils, Wooden pencils)
  11. `erasers` (Erasers, Rubber)
  12. `keyboard` (Computer keyboards)
  13. `mouse` (Computer mice, Wireless mice)
  14. `toothpaste` (Toothpaste tubes)
  15. `ball` (Sports balls, Tennis balls)
  16. `toothbrush` (Toothbrushes)
  17. `wires` (Cables, Chargers, USB wires)
- **Interactive Controls**:
  - Live Confidence Threshold Slider (0.05–0.75).
  - Resolution selector (320px Speed, 384px Balanced, 480px Accurate).
  - High-res instant snapshot capture to `snapshots/`.
  - Audio alert chime when new items enter the frame.
  - Real-time category counters, FPS gauge, and latency monitors.

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Launch Web Dashboard (Default)
```bash
python run.py --mode web
```
*Opens `http://localhost:5000` automatically in your browser.*

### Launch Standalone Desktop Window
```bash
python run.py --mode window
```

### Launch Both
```bash
python run.py --mode both
```

---

## ⌨️ Desktop Window Shortcuts
- `Q` / `ESC`: Quit application
- `S`: Save snapshot with bounding boxes & labels to `snapshots/`
- `+` / `-`: Increase / Decrease confidence threshold
- `SPACE` / `P`: Pause / Resume detection
- `C`: Toggle category statistics sidebar
