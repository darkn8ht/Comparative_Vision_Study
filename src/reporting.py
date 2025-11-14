import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def save_bar_chart(df, x_col, y_col, title, save_path):
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by=y_col, ascending=False)
    
    # --- FIX for FutureWarning ---
    ax = sns.barplot(x=x_col, y=y_col, data=df_sorted, palette="viridis", hue=x_col, legend=False)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_roc_curve(y_true, y_prob, labels, save_path):
    plt.figure(figsize=(8, 6))
    
    if len(labels) == 2: # Binary case
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
    else: # Multi-class case (One-vs-Rest)
        y_true_binarized = np.zeros_like(y_prob)
        y_true_int = y_true.astype(int)
        unique_labels = np.unique(y_true_int)
        
        for i in unique_labels:
            y_true_binarized[y_true_int == i, i] = 1
            
        for i in unique_labels:
            if i < len(labels): 
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {labels[i]} (area = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_dataset_details_table(config):
    data = []
    for task_name, task_config in config['TASKS'].items():
        if task_name == 'M1_CT_Classification':
            source = "ImageFolder (e.g., CQ500)"
        elif task_name == 'M2_LIDC_Segmentation':
            source = "LIDC-IDRI (Custom)"
        elif task_name == 'M3_Rads_Classification':
            source = "Mendeley (Lung-RADS, 5rr22hgzwr)"
        else:
            source = "Unknown"
        
        data.append({
            "Task Name": task_name,
            "Type": task_config['TYPE'],
            "Data Source": source,
            "Image Size": task_config['IMAGE_SIZE'],
            "Classes": task_config['NUM_CLASSES']
        })
    return pd.DataFrame(data)

def create_model_comparison_table(all_results):
    all_dfs = []
    for (task_name, model_name), metrics in all_results.items():
        metrics['Task'] = task_name
        metrics['Model'] = model_name
        all_dfs.append(pd.DataFrame([metrics]))
        
    if not all_dfs:
        return pd.DataFrame()

    final_report_df = pd.concat(all_dfs, ignore_index=True)
    
    final_report_df = final_report_df.rename(columns={
        'Computation Time (s)': 'Time (s)'
    })
    
    # --- MODIFIED: Updated column list ---
    cols_to_include = [
        'Task', 'Model', 'Accuracy', 'F1-Score', 'Precision', 
        'Sensitivity', 'Specificity', 'ROC AUC', 
        'CK Value', 'Time (s)', 'iou'
    ]
    
    final_cols = [col for col in cols_to_include if col in final_report_df.columns]
    final_report_df = final_report_df[final_cols]
    return final_report_df

def create_best_model_table(model_comparison_df):
    if model_comparison_df.empty:
        return pd.DataFrame()
        
    df_copy = model_comparison_df.copy()
    
    # M2 task now reports 'F1-Score', so we can simplify this
    best_models = df_copy.loc[df_copy.groupby('Task')['F1-Score'].idxmax()]
    return best_models