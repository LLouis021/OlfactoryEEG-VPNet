# =========================================================
# Utility for Olfactory EEG Project (Optuna-ready version)
# Compatible with:
# - VPCWTNN
# - vp_layer (CWTLayer)
# - trainer.py pipeline
# =========================================================

import os
import csv
import numpy as np
import torch
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# =========================================================
# Time stamp
# =========================================================
date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# =========================================================
# Loss / Acc plotting
# =========================================================
def plot_model_loss_acc(tr_l, tr_a, te_l, te_a, filename=None):

    plt.figure(figsize=(14, 6))

    # ------------------ Loss subplot ------------------
    plt.subplot(1, 2, 1)
    plt.plot(tr_l, label="Train", linewidth=1.5, color='#1f77b4')
    plt.plot(te_l, label="Test", linewidth=1.5, color='#ff7f0e')
                 
    plt.title("Model Loss", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # ------------------ Accuracy subplot ------------------
    plt.subplot(1, 2, 2)
    plt.plot(tr_a, label="Train", linewidth=1.5, color='#1f77b4')
    plt.plot(te_a, label="Test", linewidth=1.5, color='#ff7f0e')

    plt.title("Model Accuracy", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    if filename is None:
        filename = "curve.png"

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



# =========================================================
# Safe model output extraction
# =========================================================
def extract_features(model, x):
    """
    Robust feature extraction for VPCWTNN
    """
    with torch.no_grad():
        vplayer = model.vp_layers[0]
        x = vplayer(x)
        x = torch.squeeze(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
    return x


# =========================================================
# Confusion Matrix
# =========================================================
def plot_confusion_matrix(y_true, y_pred, classes, filename=None):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes)

    plt.title("Confusion Matrix", fontsize=15, pad=15)

    plt.ylabel("True Label", fontsize=13, fontweight='bold')
    plt.xlabel("Predict Label", fontsize=13, fontweight='bold')

    plt.tight_layout()

    if filename is None:
        filename = "cm.png"

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Classification report
# =========================================================
def save_classification_report(y_true, y_pred, classes, filename=None):

    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    print(report)

    if filename is None:
        filename = "report.txt"

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Date: {date_str}\n\n")
        f.write(report)


# =========================================================
# CSV logging (Optuna friendly)
# =========================================================
def write_to_log(filename, config, best_acc):

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    file_exists = os.path.exists(filename)

    header = [
        "date", "name",
        "lr", "batch_size", "epochs",
        "vp_dim", "nr",
        "weight_decay",
        "best_acc", "fold"
    ]

    row = [
        date_str,
        config.get("name", "exp"),
        config.get("lr"),
        config.get("batch_size"),
        config.get("epochs"),
        config.get("vp_dim"),
        str(config.get("nr")),
        config.get("weight_decay"),
        best_acc,
        config.get("fold_idx", 0)
    ]

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(row)


# =========================================================
# IMPORTANT: Optuna return helper
# =========================================================
def format_result(best_acc, extra=None):
    """
    Optuna uses this return structure
    """
    return {
        "best_acc": float(best_acc),
        "extra": extra
    }


# =========================================================
# Lightweight TSNE (DISABLED in Optuna loop)
# =========================================================
def plot_tsne(features, labels, class_names, filename="tsne.png"):
    from sklearn.manifold import TSNE
    import seaborn as sns
    import numpy as np
    
    # New: fault tolerance check
    if np.std(features) < 1e-6:
        print(f"[WARN] Skip t-SNE plotting: {filename}. Reason: The standard deviation of the feature vector is too small, and the model may not have converged.")
        return

    if torch.is_tensor(features):
        features = features.cpu().numpy()

    # Dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    out = tsne.fit_transform(features)

    # Plotting
    plt.figure(figsize=(14, 10))

    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Use seaborn to draw scatter plot with legend
    sns.scatterplot(
        x=out[:, 0], 
        y=out[:, 1], 
        hue=[class_names[l] for l in labels],  # Map to specific class names (A~M)
        # Core fix: force legend order to prevent seaborn from auto-reordering
        hue_order=class_names,
        # Modification 1: use 'tab20' or 'muted' palette for softer colors
        palette=sns.color_palette("tab20", len(class_names)),
        legend="full", 
        # Modification 2: fine-tune opacity for more natural overlapping edges
        alpha=0.7,
        # Modification 3: reduce point size (from 40 to ~20)
        s=20,
        # Modification 4 (most critical): remove white border from each point, key to eliminating "circle" effect
        edgecolor='none'
    )
    
    plt.title("t-SNE Visualization of Olfactory EEG Features", fontsize=16)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    
    # Move legend outside the plot to prevent obscuring data points
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Odors")
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()
