import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------
# CONFUSION MATRIX PLOT
# -------------------------
def plot_confusion_matrix(y_true, y_pred, labels=["FAKE","REAL"]):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)

    plt.title("Confusion Matrix - TraceFake AI")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved → {path}")


# -------------------------
# CLASSIFICATION REPORT
# -------------------------
def save_classification_report(y_true, y_pred):

    report = classification_report(y_true, y_pred, target_names=["FAKE","REAL"])

    path = OUTPUT_DIR / "classification_report.txt"
    with open(path, "w") as f:
        f.write(report)

    print(f"Saved → {path}")


# -------------------------
# FULL EVALUATION PIPELINE
# -------------------------
def evaluate_model(y_true, y_pred):

    print("\n=== TRACEFAKE MODEL EVALUATION ===")

    save_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)

    acc = np.mean(np.array(y_true) == np.array(y_pred))

    summary = f"""
TraceFake AI Model Report
==========================
Accuracy: {acc:.4f}

- CNN + EXIF + Forensic Fusion Model
- Fake/Real Binary Classification
"""

    path = OUTPUT_DIR / "summary.txt"
    with open(path, "w") as f:
        f.write(summary)

    print(f"Saved → {path}")