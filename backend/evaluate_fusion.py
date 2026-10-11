import torch
from torch.utils.data import DataLoader

from utils.metrics import calculate_metrics

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from feature_fusion import *


#############################################################
# Device
#############################################################

print("Using Device :", DEVICE)


#############################################################
# Load Best Fusion Model
#############################################################

print("\nLoading Best Fusion Model...")

fusion_model.load_state_dict(
    torch.load(
        "fusion/fusion_model.pth",
        map_location=DEVICE
    )
)

fusion_model.eval()

print("Fusion Model Loaded Successfully")


#############################################################
# DataLoaders
#############################################################

train_loader224 = DataLoader(
    train224,
    batch_size=16,
    shuffle=False
)

train_loader299 = DataLoader(
    train299,
    batch_size=16,
    shuffle=False
)


val_loader224 = DataLoader(
    val224,
    batch_size=16,
    shuffle=False
)

val_loader299 = DataLoader(
    val299,
    batch_size=16,
    shuffle=False
)


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
# Evaluation Function
#############################################################

def evaluate_fusion(
    loader224,
    loader299,
    dataset_name
):

    y_true = []
    y_pred = []

    correct = 0
    total = 0

    with torch.no_grad():

        for (img224, label1), (img299, label2) in zip(
            loader224,
            loader299
        ):

            #################################################
            # Move images
            #################################################

            img224 = img224.to(DEVICE)
            img299 = img299.to(DEVICE)

            labels = label1.to(DEVICE)


            #################################################
            # Extract features from all 7 CNNs
            #################################################

            features = extractor.extract(
                img224,
                img299
            )


            #################################################
            # Fusion model prediction
            #################################################

            outputs = fusion_model(features)

            _, pred = torch.max(
                outputs,
                1
            )


            #################################################
            # Accuracy
            #################################################

            total += labels.size(0)

            correct += (
                pred == labels
            ).sum().item()


            #################################################
            # Store predictions
            #################################################

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                pred.cpu().numpy()
            )


    #########################################################
    # Calculate accuracy
    #########################################################

    accuracy = correct / total


    #########################################################
    # Other metrics
    #########################################################

    _, precision, recall, f1 = calculate_metrics(
        y_true,
        y_pred
    )


    #########################################################
    # Print result
    #########################################################

    print("\n========================================")
    print(f"FEATURE FUSION - {dataset_name}")
    print("========================================")

    print(
        f"Samples   : {total}"
    )

    print(
        f"Correct   : {correct}"
    )

    print(
        f"Incorrect : {total - correct}"
    )

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    #########################################################
    # Classification Report
    #########################################################

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=dataset224.class_names
        )
    )


    #########################################################
    # Confusion Matrix
    #########################################################

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total": total,
        "y_true": y_true,
        "y_pred": y_pred
    }


#############################################################
# TRAIN
#############################################################

train_results = evaluate_fusion(
    train_loader224,
    train_loader299,
    "TRAIN"
)


#############################################################
# VALIDATION
#############################################################

val_results = evaluate_fusion(
    val_loader224,
    val_loader299,
    "VALIDATION"
)


#############################################################
# TEST
#############################################################

test_results = evaluate_fusion(
    test_loader224,
    test_loader299,
    "TEST"
)


#############################################################
# FINAL SUMMARY
#############################################################

print("\n\n")
print("================================================")
print("       FEATURE FUSION FINAL ACCURACY")
print("================================================")

print(
    f"Train Accuracy      : "
    f"{train_results['accuracy'] * 100:.2f}%"
)

print(
    f"Validation Accuracy : "
    f"{val_results['accuracy'] * 100:.2f}%"
)

print(
    f"Test Accuracy       : "
    f"{test_results['accuracy'] * 100:.2f}%"
)

print("================================================")


#############################################################
# SAVE RESULTS
#############################################################

import os

os.makedirs(
    "output",
    exist_ok=True
)


with open(
    "output/feature_fusion_results.txt",
    "w"
) as f:

    f.write("FEATURE FUSION RESULTS\n")
    f.write("=" * 50 + "\n\n")


    #########################################################
    # FINAL ACCURACY
    #########################################################

    f.write("FINAL ACCURACY\n")
    f.write("-" * 30 + "\n")

    f.write(
        f"Train Accuracy      : "
        f"{train_results['accuracy'] * 100:.2f}%\n"
    )

    f.write(
        f"Validation Accuracy : "
        f"{val_results['accuracy'] * 100:.2f}%\n"
    )

    f.write(
        f"Test Accuracy       : "
        f"{test_results['accuracy'] * 100:.2f}%\n\n"
    )


    #########################################################
    # TEST METRICS
    #########################################################

    f.write("TEST METRICS\n")
    f.write("-" * 30 + "\n")

    f.write(
        f"Accuracy  : "
        f"{test_results['accuracy'] * 100:.2f}%\n"
    )

    f.write(
        f"Precision : "
        f"{test_results['precision']:.4f}\n"
    )

    f.write(
        f"Recall    : "
        f"{test_results['recall']:.4f}\n"
    )

    f.write(
        f"F1 Score  : "
        f"{test_results['f1']:.4f}\n\n"
    )


    #########################################################
    # TEST CLASSIFICATION REPORT
    #########################################################

    f.write("TEST CLASSIFICATION REPORT\n")
    f.write("-" * 40 + "\n")

    f.write(
        classification_report(
            test_results["y_true"],
            test_results["y_pred"],
            target_names=dataset224.class_names
        )
    )


    #########################################################
    # TEST CONFUSION MATRIX
    #########################################################

    f.write("\n\nTEST CONFUSION MATRIX\n")
    f.write("-" * 40 + "\n")

    f.write(
        str(
            confusion_matrix(
                test_results["y_true"],
                test_results["y_pred"]
            )
        )
    )


print(
    "\nResults saved to "
    "output/feature_fusion_results.txt"
)