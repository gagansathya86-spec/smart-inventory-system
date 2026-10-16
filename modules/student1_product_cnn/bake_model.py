import os
import json
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "categories.json")
OUTPUT_MODEL = os.path.join(CURRENT_DIR, "yolov8s_17categories_enhanced.pt")

def bake():
    print("[Bake] Loading categories config...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    prompts = list(cfg.get("prompt_to_category", {}).keys())
    print(f"[Bake] Found {len(prompts)} prompts across {len(cfg.get('categories', []))} categories.")
    
    print("[Bake] Loading base YOLOv8s-worldv2 model...")
    model = YOLO("yolov8s-worldv2.pt")
    
    print("[Bake] Encoding open-vocabulary prompt embeddings...")
    model.set_classes(prompts)
    
    print(f"[Bake] Saving baked model to: {OUTPUT_MODEL}...")
    model.save(OUTPUT_MODEL)
    print("[Bake] Done! Model is now pre-baked and ready for fast zero-latency loading.")

if __name__ == "__main__":
    bake()
