import torch
import numpy as np

def calculate_segmentation_metrics(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).float()
    
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)
    
    TP = (preds_flat * targets_flat).sum()
    FN = targets_flat.sum() - TP
    FP = preds_flat.sum() - TP
    TN = len(targets_flat) - (TP + FP + FN)

    # --- F1-Score (Dice) ---
    f1_score = (2. * TP) / (2 * TP + FP + FN + 1e-6)
    
    # --- IoU (Jaccard) ---
    jaccard = TP / (TP + FP + FN + 1e-6)
    
    # --- Classification Metrics ---
    accuracy = (TP + TN) / (len(targets_flat) + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # --- MCC (Matthews Correlation Coefficient) ---
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-6)
    mcc = mcc_numerator / mcc_denominator
    
    return {
        'Accuracy': accuracy.item(),
        'Balanced Accuracy': balanced_accuracy.item(),
        'F1-Score': f1_score.item(),
        'Precision': precision.item(),
        'Sensitivity': sensitivity.item(),
        'Specificity': specificity.item(),
        'iou': jaccard.item(),
        'MCC': mcc.item(),
        'ROC AUC': np.nan,
        'CK Value': np.nan
    }