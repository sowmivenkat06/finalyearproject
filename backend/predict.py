import sys
import os
import torch
import numpy as np
from PIL import Image

from dataset import (
    transform_224,
    transform_299,
    preprocess_image
)
from feature_extractor import extractor
from fusion_network import FusionClassifier
from tiled_yolo import predict_tiled
from config import DEVICE, CLASS_NAMES

#############################################################
# Class Names
#############################################################

class_names = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate"
]

#############################################################
# Load Fusion Model
#############################################################

fusion_model = FusionClassifier().to(DEVICE)
checkpoint = torch.load("fusion/fusion_model.pth", map_location=DEVICE)
fusion_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
fusion_model.eval()

#############################################################
# Read Image
#############################################################

if len(sys.argv) != 2:
    print("Usage: python predict.py image.jpg")
    sys.exit()

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: File not found: {image_path}")
    sys.exit(1)

full_image = Image.open(image_path).convert("RGB")
w_img, h_img = full_image.size

yolo_path = "yolo_cells.pt"
detections = []

if os.path.exists(yolo_path):
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_path)
        detections = predict_tiled(
            full_image,
            yolo_model,
            tile_size=640,
            overlap=0.2,
            conf=0.20,
            iou_thresh=0.45
        )
        print(f"✓ Tiled YOLO cell detection: {len(detections)} cell(s) localized.")
    except Exception as e:
        print(f"YOLO detection skipped: {e}")

if detections:
    print("\n==============================================")
    print("MULTI-CELL LOCALIZATION & CLASSIFICATION")
    print("==============================================")
    
    cell_predictions = []
    overall_probs = np.zeros(len(class_names), dtype=np.float32)

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['box']
        pad = 15
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w_img, x2 + pad)
        cy2 = min(h_img, y2 + pad)

        crop = full_image.crop((cx1, cy1, cx2, cy2))
        preprocessed = preprocess_image(crop, auto_crop=False)

        img224 = transform_224(preprocessed).unsqueeze(0).to(DEVICE)
        img299 = transform_299(preprocessed).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = extractor.extract(img224, img299)
            output = fusion_model(features)
            probs = torch.softmax(output, dim=1).cpu().squeeze().numpy()

        overall_probs += probs
        pred_idx = np.argmax(probs)
        cell_class = class_names[pred_idx]
        conf = probs[pred_idx]

        cell_predictions.append((cell_class, conf, probs))
        print(f"Cell #{i:02d} [{x1:4d},{y1:4d},{x2:4d},{y2:4d}] -> {cell_class} ({conf*100:.1f}%) [YOLO conf: {det['conf']*100:.1f}%]")

    overall_probs /= len(detections)
    overall_pred_idx = np.argmax(overall_probs)

    print("\n==============================================")
    print("SLIDE SUMMARY PREDICTION")
    print("==============================================")
    print("Total Cells Detected :", len(detections))
    print("Primary Slide Class  :", class_names[overall_pred_idx])
    print(f"Overall Confidence   : {overall_probs[overall_pred_idx]*100:.2f}%")
    print("\nAverage Class Probabilities:")
    for idx, cls in enumerate(class_names):
        print(f"  {cls}: {overall_probs[idx]*100:.2f}%")

else:
    print("✓ Running single-image preprocessing (CLAHE + Median + Otsu).")
    image = preprocess_image(full_image, auto_crop=True)
    img224 = transform_224(image).unsqueeze(0).to(DEVICE)
    img299 = transform_299(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = extractor.extract(img224, img299)
        output = fusion_model(features)
        probability = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probability, 1)

        print("\nClass Probabilities:")
        for i, cls in enumerate(class_names):
            print(f"  {cls}: {probability[0][i].item()*100:.2f}%")

    print("\n==============================")
    print("Prediction Result")
    print("==============================")
    print("Predicted Class :", class_names[prediction.item()])
    print(f"Confidence      : {confidence.item()*100:.2f}%")

