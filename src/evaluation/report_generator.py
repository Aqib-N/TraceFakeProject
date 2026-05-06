"""
TraceFake AI — Comprehensive Evaluation & Reporting
Generates all required assignment deliverables
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------
# CONFUSION MATRIX PLOT
# -------------------------
def plot_confusion_matrix(y_true, y_pred, labels=["FAKE","REAL"], save_name="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(len(labels)):
        for j in range(len(labels)):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.75, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, color='gray')
    
    plt.title("Confusion Matrix — TraceFake AI", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved confusion matrix → {path}")
    return cm


# -------------------------
# CLASSIFICATION REPORT
# -------------------------
def save_classification_report(y_true, y_pred, save_name="classification_report.txt"):
    report = classification_report(y_true, y_pred, target_names=["FAKE","REAL"])
    
    # Compute additional metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    extended_report = f"""
{report}

Additional Metrics:
-------------------
Specificity (TNR): {specificity:.4f}
Negative Predictive Value: {npv:.4f}
True Positives: {tp}
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
"""
    
    path = OUTPUT_DIR / save_name
    with open(path, "w") as f:
        f.write("TraceFake AI — Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(extended_report)
    
    print(f"📝 Saved classification report → {path}")
    return extended_report


# -------------------------
# ROC CURVE
# -------------------------
def plot_roc_curve(y_true, y_scores, save_name="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#00D4E8', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve — TraceFake AI', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved ROC curve → {path}")
    return roc_auc


# -------------------------
# FULL EVALUATION PIPELINE
# -------------------------
def evaluate_model(y_true, y_pred, y_scores=None, model_name="TraceFake AI"):
    print(f"\n{'='*60}")
    print(f"    {model_name} — COMPREHENSIVE EVALUATION")
    print(f"{'='*60}")
    
    # 1. Classification report
    save_classification_report(y_true, y_pred)
    
    # 2. Confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # 3. ROC curve (if scores provided)
    if y_scores is not None:
        roc_auc = plot_roc_curve(y_true, y_scores)
    else:
        roc_auc = None
    
    # 4. Summary statistics
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    summary = f"""
{model_name} Evaluation Summary
{'='*60}
Accuracy:  {acc:.4f}
AUC:       {roc_auc:.4f if roc_auc else 'N/A'}

Model Architecture:
- CNN: EfficientNetB0 (Transfer Learning)
- EXIF: XGBoost with 10 forensic features
- Forensic: ELA + Noise + JPEG + Chromatic fusion
- Fusion: Optimized weighted ensemble

Dataset:
- Binary Classification: FAKE vs REAL
- Image Size: 224×224
- Preprocessing: Rescaling, Augmentation
"""
    
    path = OUTPUT_DIR / "evaluation_summary.txt"
    with open(path, "w") as f:
        f.write(summary)
    
    print(f"\n📋 Saved evaluation summary → {path}")
    print(f"\n✅ Evaluation complete! All reports in {OUTPUT_DIR}/")
    
    return {
        'accuracy': acc,
        'auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }