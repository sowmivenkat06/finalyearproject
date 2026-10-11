import torch
import torch.nn.functional as F
from utils.metrics import calculate_metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from feature_extractor import *
from dataset import *

from torchvision.models import *

#########################################################
# Load Dataset
#########################################################

dataset224 = CervicalDataset(
    DATASET_PATH,
    transform_224
)

dataset299 = CervicalDataset(
    DATASET_PATH,
    transform_299
)
#########################################################
# Load Trained Models
#########################################################

NUM_CLASSES = 5

# ---------------- ResNet50 ---------------- #

resnet50_model = resnet50(weights=None)
resnet50_model.fc = torch.nn.Linear(
    resnet50_model.fc.in_features,
    NUM_CLASSES
)

resnet50_model.load_state_dict(
    torch.load(
        "models/ResNet50.pth",
        map_location=DEVICE
    )
)

resnet50_model.to(DEVICE)
resnet50_model.eval()


# ---------------- ResNet101 ---------------- #

resnet101_model = resnet101(weights=None)
resnet101_model.fc = torch.nn.Linear(
    resnet101_model.fc.in_features,
    NUM_CLASSES
)

resnet101_model.load_state_dict(
    torch.load(
        "models/ResNet101.pth",
        map_location=DEVICE
    )
)

resnet101_model.to(DEVICE)
resnet101_model.eval()


# ---------------- ResNet152 ---------------- #

resnet152_model = resnet152(weights=None)
resnet152_model.fc = torch.nn.Linear(
    resnet152_model.fc.in_features,
    NUM_CLASSES
)

resnet152_model.load_state_dict(
    torch.load(
        "models/ResNet152.pth",
        map_location=DEVICE
    )
)

resnet152_model.to(DEVICE)
resnet152_model.eval()


# ---------------- AlexNet ---------------- #

alexnet_model = alexnet(weights=None)
alexnet_model.classifier[6] = torch.nn.Linear(
    alexnet_model.classifier[6].in_features,
    NUM_CLASSES
)

alexnet_model.load_state_dict(
    torch.load(
        "models/AlexNet.pth",
        map_location=DEVICE
    )
)

alexnet_model.to(DEVICE)
alexnet_model.eval()


# ---------------- VGG16 ---------------- #

vgg16_model = vgg16(weights=None)
vgg16_model.classifier[6] = torch.nn.Linear(
    vgg16_model.classifier[6].in_features,
    NUM_CLASSES
)

vgg16_model.load_state_dict(
    torch.load(
        "models/VGG16.pth",
        map_location=DEVICE
    )
)

vgg16_model.to(DEVICE)
vgg16_model.eval()


# ---------------- GoogleNet ---------------- #

googlenet_model = googlenet(
    weights=None,
    aux_logits=False
)

googlenet_model.fc = torch.nn.Linear(
    googlenet_model.fc.in_features,
    NUM_CLASSES
)

googlenet_model.load_state_dict(
    torch.load(
        "models/GoogleNet.pth",
        map_location=DEVICE
    ),
    strict=False
)

googlenet_model.to(DEVICE)
googlenet_model.eval()


# ---------------- InceptionV3 ---------------- #

inception_model = inception_v3(
    weights=None,
    aux_logits=False
)

inception_model.fc = torch.nn.Linear(
    inception_model.fc.in_features,
    NUM_CLASSES
)

inception_model.load_state_dict(
    torch.load(
        "models/InceptionV3.pth",
        map_location=DEVICE
    ),
    strict=False
)

inception_model.to(DEVICE)
inception_model.eval()

print("All Models Loaded Successfully")

#########################################################
# Same Test Split
#########################################################

from torch.utils.data import DataLoader, Subset
import numpy as np

indices = np.random.permutation(len(dataset224))

train_end = int(0.70 * len(indices))
val_end = int(0.85 * len(indices))

test_idx = indices[val_end:]

test224 = Subset(dataset224, test_idx)
test299 = Subset(dataset299, test_idx)

#########################################################
# Test Loader
#########################################################

test_loader224 = DataLoader(
    test224,
    batch_size=16,
    shuffle=False
)

test_loader299 = DataLoader(
    test299,
    batch_size=16,
    shuffle=False
)
#########################################################
# Decision Fusion (Soft Voting)
#########################################################

y_true = []
y_pred = []

with torch.no_grad():

    for (img224, label1), (img299, label2) in zip(
        test_loader224,
        test_loader299
    ):

        img224 = img224.to(DEVICE)
        img299 = img299.to(DEVICE)

        labels = label1.to(DEVICE)

        ###################################################
        # Individual Model Predictions
        ###################################################

        p1 = F.softmax(resnet50_model(img224), dim=1)
        p2 = F.softmax(resnet101_model(img224), dim=1)
        p3 = F.softmax(resnet152_model(img224), dim=1)

        p4 = F.softmax(alexnet_model(img224), dim=1)

        p5 = F.softmax(vgg16_model(img224), dim=1)

        p6 = F.softmax(googlenet_model(img224), dim=1)

        p7 = F.softmax(inception_model(img299), dim=1)

        ###################################################
        # Soft Voting
        ###################################################

        avg_prob = (

            p1 +
            p2 +
            p3 +
            p4 +
            p5 +
            p6 +
            p7

        ) / 7

        prediction = torch.argmax(avg_prob, dim=1)

        y_true.extend(labels.cpu().numpy())

        y_pred.extend(prediction.cpu().numpy())

#########################################################
# Evaluation
#########################################################

accuracy, precision, recall, f1 = calculate_metrics(
    y_true,
    y_pred
)

print("\n==============================")
print("DECISION FUSION RESULTS")
print("==============================")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=dataset224.class_names
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)
import os

os.makedirs("output", exist_ok=True)

with open("output/decision_fusion_results.txt", "w") as f:

    f.write("DECISION FUSION RESULTS\n")
    f.write("=" * 40 + "\n")

    f.write(f"Accuracy  : {accuracy*100:.2f}%\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n\n")

    f.write("Classification Report\n\n")
    f.write(classification_report(
        y_true,
        y_pred,
        target_names=dataset224.class_names
    ))

    f.write("\n\nConfusion Matrix\n")
    f.write(str(confusion_matrix(y_true, y_pred)))

print("Results saved to output/decision_fusion_results.txt")