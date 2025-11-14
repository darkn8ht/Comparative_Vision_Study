import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, cohen_kappa_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef, log_loss
)
from statsmodels.stats.contingency_tables import mcnemar

def calculate_all_metrics(y_true, y_pred, y_prob, labels):
    is_multi_class = len(labels) > 2
    average_type = 'weighted' if is_multi_class else 'binary'
    
    if is_multi_class:
        auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    else:
        auc_score = roc_auc_score(y_true, y_prob[:, 1])

    if is_multi_class:
        positive_class_index = len(labels) - 1
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
        tp = cm[positive_class_index, positive_class_index]
        fn = np.sum(cm[positive_class_index, :]) - tp
        fp = np.sum(cm[:, positive_class_index]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
    else:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (1, 1): 
            if np.unique(y_true)[0] == 1:
                tp, fn, fp, tn = cm[0,0], 0, 0, 0
            else:
                tp, fn, fp, tn = 0, 0, 0, cm[0,0]
        elif cm.shape == (0, 0) or len(cm.ravel()) < 4:
            tp, fn, fp, tn = 0, 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average_type, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average=average_type, zero_division=0),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'ROC AUC': auc_score,
        'CK Value': cohen_kappa_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Log Loss': log_loss(y_true, y_prob)
    }
    return metrics