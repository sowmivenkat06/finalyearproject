"""Generate reproducible, real test metrics for each saved backbone checkpoint.
This evaluates only; it never alters checkpoint weights or retrains a model.
"""
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision.models import alexnet, googlenet, inception_v3, resnet50, resnet101, resnet152, vgg16

from config import CLASS_NAMES, DATASET_PATH, DEVICE, NUM_CLASSES
from dataset import CervicalDataset, transform_224, transform_299

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 20260819

def build_models():
    specs = {"ResNet50": (resnet50, "fc", False), "ResNet101": (resnet101, "fc", False), "ResNet152": (resnet152, "fc", False), "AlexNet": (alexnet, "classifier", False), "VGG16": (vgg16, "classifier", False), "GoogleNet": (googlenet, "fc", False), "InceptionV3": (inception_v3, "fc", True)}
    models = {}
    for name, (factory, head, is_inception) in specs.items():
        if name == "GoogleNet": model = factory(weights=None, aux_logits=False)
        elif is_inception: model = factory(weights=None, aux_logits=True)
        else: model = factory(weights=None)
        if head == "classifier": model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        else: model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        if is_inception: model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", f"{name}.pth"), map_location=DEVICE))
        models[name] = model.to(DEVICE).eval()
    return models

def metric_record(y_true, probabilities):
    predicted = probabilities.argmax(axis=1)
    try: auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")
    except ValueError: auc = None
    return {"accuracy": accuracy_score(y_true, predicted), "precision_weighted": precision_score(y_true, predicted, average="weighted", zero_division=0), "recall_weighted": recall_score(y_true, predicted, average="weighted", zero_division=0), "f1_weighted": f1_score(y_true, predicted, average="weighted", zero_division=0), "auc_ovr_weighted": auc, "confusion_matrix": confusion_matrix(y_true, predicted).tolist(), "classification_report": classification_report(y_true, predicted, target_names=CLASS_NAMES, output_dict=True, zero_division=0)}

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dataset224 = CervicalDataset(DATASET_PATH, transform_224); dataset299 = CervicalDataset(DATASET_PATH, transform_299)
    indices = np.random.permutation(len(dataset224)); test_indices = indices[int(.85 * len(indices)):]
    loader224 = DataLoader(Subset(dataset224, test_indices), batch_size=8, shuffle=False); loader299 = DataLoader(Subset(dataset299, test_indices), batch_size=8, shuffle=False)
    models = build_models(); truth = []; scores = {name: [] for name in models}
    with torch.no_grad():
        for (x224, labels), (x299, _) in zip(loader224, loader299):
            x224, x299 = x224.to(DEVICE), x299.to(DEVICE); truth.extend(labels.tolist())
            for name, model in models.items(): scores[name].append(torch.softmax(model(x299 if name == "InceptionV3" else x224), dim=1).cpu().numpy())
    output = {"seed": SEED, "test_samples": len(truth), "class_names": CLASS_NAMES, "models": {name: metric_record(truth, np.concatenate(parts)) for name, parts in scores.items()}}
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "output", "model_evaluation.json"), "w", encoding="utf-8") as f: json.dump(output, f, indent=2)
    print(json.dumps({name: round(data["accuracy"] * 100, 2) for name, data in output["models"].items()}, indent=2))

if __name__ == "__main__": main()
