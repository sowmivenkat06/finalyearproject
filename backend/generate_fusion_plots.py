import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from feature_fusion import (
    fusion_model,
    extractor,
    test224,
    test299,
    dataset224,
    DEVICE,
    NUM_CLASSES
)

def generate_plots():
    print("=" * 60)
    print("GENERATING FEATURE LEVEL FUSION PERFORMANCE PLOTS")
    print("=" * 60)

    # 1. Load trained model weights
    model_path = "fusion/fusion_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    fusion_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
    fusion_model.eval()

    test_loader224 = DataLoader(test224, batch_size=32, shuffle=False)
    test_loader299 = DataLoader(test299, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []
    y_probs = []

    print(f"Evaluating Feature Fusion model on test samples ({len(test224)} samples)...", flush=True)

    with torch.no_grad():
        for batch_idx, ((img224, label1), (img299, label2)) in enumerate(zip(test_loader224, test_loader299)):
            img224 = img224.to(DEVICE)
            img299 = img299.to(DEVICE)
            labels = label1.to(DEVICE)

            features = extractor.extract(img224, img299)
            outputs = fusion_model(features)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            print(f"Processed batch {batch_idx+1}/{len(test_loader224)}", flush=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    class_names = [c.replace("im_", "") for c in dataset224.class_names]
    num_classes = len(class_names)

    output_dir = "output"
    static_img_dir = os.path.join("static", "img")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)

    # Set overall aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.sans-serif': 'Inter, Roboto, Arial, sans-serif',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 1.2
    })

    # =========================================================
    # 1. CONFUSION MATRIX HEATMAP
    # =========================================================
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    
    # Custom annotation text showing count and percentage
    annot_matrix = np.empty((num_classes, num_classes), dtype=object)
    for i in range(num_classes):
        for j in range(num_classes):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"}
    )

    ax.set_title("Feature Level Fusion Model — Confusion Matrix", fontsize=14, fontweight="bold", pad=15, color="#1e293b")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=10, color="#334155")
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold", labelpad=10, color="#334155")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    cm_output_path = os.path.join(output_dir, "feature_fusion_confusion_matrix.png")
    cm_static_path = os.path.join(static_img_dir, "feature_fusion_confusion_matrix.png")
    plt.savefig(cm_output_path, dpi=300, bbox_inches="tight")
    plt.savefig(cm_static_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Confusion Matrix graph to {cm_output_path} and {cm_static_path}")

    # =========================================================
    # 2. STANDALONE ROC CURVES
    # =========================================================
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
    
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i in range(num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        ax.plot(
            fpr_dict[i],
            tpr_dict[i],
            color=colors[i % len(colors)],
            lw=2.5,
            label=f"{class_names[i]} (AUC = {roc_auc_dict[i]:.4f})"
        )

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(
        fpr_micro,
        tpr_micro,
        color="#0f172a",
        linestyle="--",
        lw=2.5,
        label=f"Micro-Average (AUC = {roc_auc_micro:.4f})"
    )

    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.5, label="Random Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Feature Level Fusion Model — Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()

    roc_output_path = os.path.join(output_dir, "feature_fusion_roc_curves.png")
    roc_static_path = os.path.join(static_img_dir, "feature_fusion_roc_curves.png")
    plt.savefig(roc_output_path, dpi=300, bbox_inches="tight")
    plt.savefig(roc_static_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ROC Curves graph to {roc_output_path} and {roc_static_path}")

    # =========================================================
    # 3. COMPREHENSIVE PERFORMANCE CURVES DASHBOARD (2x2)
    # =========================================================
    fig, axs = plt.subplots(2, 2, figsize=(16, 13), dpi=300)

    # Panel A: ROC Curves
    for i in range(num_classes):
        axs[0, 0].plot(
            fpr_dict[i],
            tpr_dict[i],
            color=colors[i % len(colors)],
            lw=2.2,
            label=f"{class_names[i]} ({roc_auc_dict[i]:.4f})"
        )
    axs[0, 0].plot(fpr_micro, tpr_micro, color="#0f172a", linestyle="--", lw=2, label=f"Micro Avg ({roc_auc_micro:.4f})")
    axs[0, 0].plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.5)
    axs[0, 0].set_title("A) ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold", color="#0f172a")
    axs[0, 0].set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    axs[0, 0].set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    axs[0, 0].legend(loc="lower right", fontsize=9, frameon=True)

    # Panel B: Precision-Recall Curves
    precision_dict = {}
    recall_dict = {}
    ap_dict = {}

    for i in range(num_classes):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        ap_dict[i] = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        axs[0, 1].plot(
            recall_dict[i],
            precision_dict[i],
            color=colors[i % len(colors)],
            lw=2.2,
            label=f"{class_names[i]} (AP = {ap_dict[i]:.4f})"
        )
    axs[0, 1].set_title("B) Precision-Recall Curves", fontsize=13, fontweight="bold", color="#0f172a")
    axs[0, 1].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    axs[0, 1].set_ylabel("Precision (PPV)", fontsize=11, fontweight="bold")
    axs[0, 1].legend(loc="lower left", fontsize=9, frameon=True)

    # Panel C: Summary Performance Metrics Bar Chart
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    metric_values = [acc * 100, prec * 100, rec * 100, f1 * 100, roc_auc_micro * 100]
    bar_colors = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#1d4ed8"]

    bars = axs[1, 0].bar(metric_names, metric_values, color=bar_colors, width=0.55, edgecolor="#1e40af", linewidth=1.2)
    axs[1, 0].set_ylim([90, 101])
    axs[1, 0].set_title("C) Overall Test Performance (%)", fontsize=13, fontweight="bold", color="#0f172a")
    axs[1, 0].set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        axs[1, 0].annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1e293b"
        )

    # Panel D: Per-Class F1-Scores
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    class_f1_scores = [report[c]["f1-score"] * 100 for c in class_names]

    bars2 = axs[1, 1].barh(class_names, class_f1_scores, color=colors, height=0.55, edgecolor="#334155", linewidth=1.2)
    axs[1, 1].set_xlim([90, 101])
    axs[1, 1].set_title("D) Per-Class F1-Scores (%)", fontsize=13, fontweight="bold", color="#0f172a")
    axs[1, 1].set_xlabel("F1-Score (%)", fontsize=11, fontweight="bold")
    for bar in bars2:
        width = bar.get_width()
        axs[1, 1].annotate(
            f"{width:.2f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e293b"
        )

    plt.suptitle("Feature Level Fusion Model — Comprehensive Performance Dashboard", fontsize=16, fontweight="bold", y=0.98, color="#0f172a")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    perf_output_path = os.path.join(output_dir, "feature_fusion_performance_curves.png")
    perf_static_path = os.path.join(static_img_dir, "feature_fusion_performance_curves.png")
    plt.savefig(perf_output_path, dpi=300, bbox_inches="tight")
    plt.savefig(perf_static_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Comprehensive Performance Curves graph to {perf_output_path} and {perf_static_path}")

    print("\nFeature Fusion performance plots generated successfully!")

if __name__ == "__main__":
    generate_plots()
