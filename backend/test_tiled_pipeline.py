import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set backend path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from config import CLASS_NAMES, DEVICE
from dataset import transform_224, transform_299, preprocess_image
from feature_extractor import extractor
from fusion_network import FusionClassifier
from tiled_yolo import predict_tiled
from ultralytics import YOLO

def run_tiled_pipeline_test():
    print("=" * 60)
    print("CELL LOCALIZATION + FUSION CLASSIFICATION PIPELINE TEST")
    print("=" * 60)

    # 1. Load YOLO model
    yolo_path = os.path.join(APP_DIR, "yolo_cells.pt")
    if not os.path.exists(yolo_path):
        print(f"Error: YOLO weights not found at {yolo_path}")
        return
    print(f"Loading YOLO cell detector from: {yolo_path}")
    yolo_model = YOLO(yolo_path)

    # 2. Load Feature Attention Fusion Model
    fusion_path = os.path.join(APP_DIR, "fusion", "fusion_model.pth")
    if not os.path.exists(fusion_path):
        print(f"Error: Fusion model checkpoint not found at {fusion_path}")
        return
    print(f"Loading Feature Attention Fusion Classifier from: {fusion_path}")
    fusion_model = FusionClassifier().to(DEVICE)
    checkpoint = torch.load(fusion_path, map_location=DEVICE)
    fusion_model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    )
    fusion_model.eval()

    # Output directory for annotated test results
    out_dir = os.path.join(APP_DIR, "outputs", "tiled_detections")
    os.makedirs(out_dir, exist_ok=True)

    # Test images across 5 classes
    dataset_dir = os.path.join(APP_DIR, "dataset")
    test_cases = []

    for cls_name in CLASS_NAMES:
        cls_folder = os.path.join(dataset_dir, cls_name)
        if not os.path.exists(cls_folder):
            cls_folder = os.path.join(dataset_dir, f"im_{cls_name}")

        if os.path.exists(cls_folder):
            # Find first full image (.bmp or .jpg)
            bmps = [f for f in sorted(os.listdir(cls_folder)) if f.lower().endswith((".bmp", ".jpg", ".jpeg")) and not f.startswith(".")]
            if bmps:
                test_cases.append((cls_name, os.path.join(cls_folder, bmps[0])))

    if not test_cases:
        print("No test images found in dataset folder.")
        # Fallback to test.jpg if present
        if os.path.exists(os.path.join(APP_DIR, "test.jpg")):
            test_cases.append(("test_sample", os.path.join(APP_DIR, "test.jpg")))

    print(f"\nFound {len(test_cases)} full microscope test images:")
    for cls, path in test_cases:
        print(f" - {cls}: {os.path.basename(path)}")

    total_cells_detected = 0

    for cls_name, img_path in test_cases:
        print("\n" + "-" * 50)
        print(f"Processing: {cls_name} ({os.path.basename(img_path)})")
        
        full_img = Image.open(img_path).convert("RGB")
        w_img, h_img = full_img.size
        print(f"Image resolution: {w_img}x{h_img}")

        # STEP 3: Tiled YOLO Detection
        start_time = time.time()
        detections = predict_tiled(
            full_img,
            yolo_model,
            tile_size=640,
            overlap=0.2,
            conf=0.20,
            iou_thresh=0.45
        )
        det_time = time.time() - start_time
        print(f"YOLO Tiled Detection: {len(detections)} cells found in {det_time:.2f}s")

        total_cells_detected += len(detections)

        # Prepare canvas for annotations
        annotated_img = full_img.copy()
        draw = ImageDraw.Draw(annotated_img)

        # Colors for visualization
        color_map = {
            "im_Dyskeratotic": "#E63946",
            "im_Koilocytotic": "#F4A261",
            "im_Metaplastic": "#2A9D8F",
            "im_Parabasal": "#457B9D",
            "im_Superficial-Intermediate": "#1D3557"
        }

        cell_results = []

        # Process each detected cell crop
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['box']
            det_conf = det['conf']

            # Add padding around detected cell box
            pad = 15
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w_img, x2 + pad)
            cy2 = min(h_img, y2 + pad)

            # STEP 5: Crop cell
            cell_crop = full_img.crop((cx1, cy1, cx2, cy2))

            # STEP 6: Preprocess crop (CLAHE + Median + Otsu with auto_crop=False)
            preprocessed_crop = preprocess_image(cell_crop, auto_crop=False)

            # STEP 7: Transform & Extract 17,408 features
            img224 = transform_224(preprocessed_crop).unsqueeze(0).to(DEVICE)
            img299 = transform_299(preprocessed_crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                features = extractor.extract(img224, img299) # 17,408 features
                # STEP 8: Feature Attention Fusion Classifier
                output = fusion_model(features)
                probs = torch.softmax(output, dim=1).cpu().squeeze().numpy()

            pred_idx = np.argmax(probs)
            pred_class = CLASS_NAMES[pred_idx]
            pred_conf = float(probs[pred_idx])

            cell_results.append({
                'cell_num': i,
                'box': [x1, y1, x2, y2],
                'yolo_conf': det_conf,
                'pred_class': pred_class,
                'pred_conf': pred_conf,
                'probabilities': {cls: float(probs[j]) for j, cls in enumerate(CLASS_NAMES)}
            })

            print(f"  Cell #{i:02d} @ [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}] -> {pred_class} ({pred_conf*100:.1f}%) [YOLO: {det_conf*100:.1f}%]")

            # STEP 9: Draw box & label on full image
            color = color_map.get(pred_class, "#FF0000")
            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Label
            label_text = f"#{i} {pred_class.replace('im_', '')} {pred_conf*100:.0f}%"
            
            # Text background pill
            text_bbox = draw.textbbox((x1, max(0, y1 - 20)), label_text)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, max(0, y1 - 20)), label_text, fill="white")

        # Save annotated image
        save_filename = f"tiled_{cls_name}_{os.path.basename(img_path)}"
        if not save_filename.lower().endswith('.jpg'):
            save_filename = os.path.splitext(save_filename)[0] + '.jpg'
        save_path = os.path.join(out_dir, save_filename)
        annotated_img.save(save_path, "JPEG", quality=92)
        print(f"✓ Saved annotated detection image to: {save_path}")

    print("\n" + "=" * 60)
    print(f"PIPELINE TEST COMPLETE! Total cells localized across images: {total_cells_detected}")
    print(f"Results saved in: {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    run_tiled_pipeline_test()
