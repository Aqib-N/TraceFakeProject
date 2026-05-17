import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
from pathlib import Path
from config import REPORT_DIR

OUTPUT_DIR = REPORT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# CONFUSION MATRIX PLOT

def plot_confusion_matrix(y_true, y_pred, labels=["FAKE","REAL","UNCERTAIN"], save_name="confusion_matrix.png"):
    # Filter out UNCERTAIN from confusion matrix if not present in predictions
    present = sorted(set(y_true) | set(y_pred))
    active_labels = [l for l in labels if l in present]
    cm = confusion_matrix(y_true, y_pred, labels=active_labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=active_labels,
                yticklabels=active_labels,
                cbar_kws={'label': 'Count'})

    # Add percentage annotations
    total = cm.sum()
    for i in range(len(active_labels)):
        for j in range(len(active_labels)):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.75, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, color='gray')

    plt.title("Confusion Matrix — TraceFake AI", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()

    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved confusion matrix → {path}")
    return cm



# CLASSIFICATION REPORT

def save_classification_report(y_true, y_pred, save_name="classification_report.txt"):
    present = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, target_names=present, labels=present)

    # Compute additional metrics (binary only when no UNCERTAIN)
    if "UNCERTAIN" not in present:
        cm = confusion_matrix(y_true, y_pred, labels=present)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        extra = f"""
Additional Metrics:
-------------------
Specificity (TNR): {specificity:.4f}
Negative Predictive Value: {npv:.4f}
True Positives: {tp}
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
"""
    else:
        extra = "\nNote: UNCERTAIN class present — binary metrics (TNR, NPV) skipped.\n"

    extended_report = f"\n{report}{extra}"
    
    path = OUTPUT_DIR / save_name
    with open(path, "w") as f:
        f.write("TraceFake AI — Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(extended_report)
    
    print(f"📝 Saved classification report → {path}")
    return extended_report



# ROC CURVE

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
    plt.tight_layout()

    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved ROC curve → {path}")
    return roc_auc



# CONFIDENCE DISTRIBUTION

def plot_confidence_distribution(y_scores, y_true, save_name="confidence_distribution.png"):
    """Plot histogram of model confidence scores split by true class."""
    y_scores = np.array(y_scores)
    y_true   = np.array(y_true)

    plt.figure(figsize=(8, 5))
    for label, color in [(1, '#00D4E8'), (0, '#FF6B6B')]:
        mask = y_true == label
        name = "REAL" if label == 1 else "FAKE"
        plt.hist(y_scores[mask], bins=30, alpha=0.6, color=color, label=name, edgecolor='none')

    plt.xlabel("Confidence Score (P(REAL))", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Confidence Distribution — TraceFake AI", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📉 Saved confidence distribution → {path}")



# FULL EVALUATION PIPELINE

def evaluate_model(y_true, y_pred, y_scores=None, model_name="TraceFake AI"):
    print(f"\n{'='*60}")
    print(f"    {model_name} — COMPREHENSIVE EVALUATION")
    print(f"{'='*60}")
    
    # 1. Classification report
    save_classification_report(y_true, y_pred)
    
    # 2. Confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # 3. ROC curve + confidence distribution (if scores provided)
    if y_scores is not None:
        roc_auc = plot_roc_curve(y_true, y_scores)
        plot_confidence_distribution(y_scores, y_true)
    else:
        roc_auc = None
    
    # 4. Summary statistics
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"

    summary = f"""
{model_name} Evaluation Summary
{'='*60}
Accuracy:  {acc:.4f}
AUC:       {auc_str}

Model Architecture:
- CNN: EfficientNetB0 (Transfer Learning)
- EXIF: XGBoost with 12 forensic features
- Forensic: ELA + Noise + JPEG + Chromatic + FFT fusion
- Fusion: Optimized weighted ensemble

Dataset:
- Binary Classification: FAKE vs REAL (+ UNCERTAIN)
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