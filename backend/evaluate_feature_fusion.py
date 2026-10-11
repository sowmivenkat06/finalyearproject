import torch
from utils.metrics import calculate_metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from feature_fusion import *

#############################################################
# Load Best Fusion Model
#############################################################

fusion_model.load_state_dict(
    torch.load(
        "fusion/fusion_model.pth",
        map_location=DEVICE
    )
)

fusion_model.eval()

#############################################################
# Test DataLoader
#############################################################

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

#############################################################
# Prediction
#############################################################

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

        #####################################################
        # Extract Features
        #####################################################

        features = extractor.extract(

            img224,
            img299

        )

        #####################################################
        # Fusion Prediction
        #####################################################

        outputs = fusion_model(features)

        _, pred = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())

        y_pred.extend(pred.cpu().numpy())

#############################################################
# Metrics
#############################################################

accuracy, precision, recall, f1 = calculate_metrics(
    y_true,
    y_pred
)

print("\n==============================")
print("FEATURE FUSION RESULTS")
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

with open("output/feature_fusion_results.txt", "w") as f:

    f.write("FEATURE FUSION RESULTS\n")
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

print("Results saved to output/feature_fusion_results.txt")

try:
    from generate_fusion_plots import generate_plots
    generate_plots()
except Exception as e:
    print(f"Plot generation skipped: {e}")