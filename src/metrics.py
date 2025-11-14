import torch
import numpy as np

def calculate_segmentation_metrics(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).float()
    
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)
    
    # Calculate components
    TP = (preds_flat * targets_flat).sum()
    FN = targets_flat.sum() - TP
    FP = preds_flat.sum() - TP
    TN = len(targets_flat) - (TP + FP + FN)

    # --- F1-Score (Dice) ---
    dice_score = (2. * TP) / (2 * TP + FP + FN + 1e-6)
    
    # --- IoU (Jaccard) ---
    jaccard = TP / (TP + FP + FN + 1e-6)
    
    # --- Classification Metrics ---
    accuracy = (TP + TN) / (len(targets_flat) + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6) # Same as recall
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    
    return {
        'Accuracy': accuracy.item(),
        'F1-Score': dice_score.item(), # Use F1-Score to match other tasks
        'Precision': precision.item(),
        'Sensitivity': sensitivity.item(),
        'Specificity': specificity.item(),
        'dice': dice_score.item(), # Keep dice for internal val logging
        'iou': jaccard.item(),
        'ROC AUC': np.nan, # Not calculated for segmentation
        'CK Value': np.nan # Not calculated for segmentation
    }