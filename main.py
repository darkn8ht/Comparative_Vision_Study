import torch
import os
import random
import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import time
from src.data_loader import get_data_loaders
from src.train_segmentation import run_segmentation_task 
from src.model_architectures import get_model
from src.training import train_model
from src.evaluation.evaluation import calculate_all_metrics
from src.reporting import (
    save_confusion_matrix, save_roc_curve, 
    create_dataset_details_table, create_model_comparison_table, 
    create_best_model_table, save_bar_chart
)
from torch import nn, optim

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_classification_model(model, loader, device, labels, report_dir, model_name):
    model.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []

    with torch.no_grad():
        for data, target in tqdm(loader, desc="[TEST]"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1).cpu().numpy()
            preds = torch.argmax(output, dim=1).cpu().numpy()
            
            y_true_list.extend(target.cpu().numpy())
            y_pred_list.extend(preds)
            y_prob_list.extend(probs)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    metrics = calculate_all_metrics(y_true, y_pred, y_prob, labels)
    
    save_confusion_matrix(y_true, y_pred, labels, 
                          os.path.join(report_dir, f"{model_name}_confusion_matrix.png"))
    save_roc_curve(y_true, y_prob, labels, 
                   os.path.join(report_dir, f"{model_name}_roc_curve.png"))
    
    return metrics, y_true, y_pred, y_prob


def run_classification_task(task_name, task_config, train_loader, val_loader, test_loader, device, eval_only=False):
    results = {}
    num_classes = task_config['NUM_CLASSES']
    base_cfg = task_config['BASE_CONFIG']
    report_dir = os.path.join(base_cfg['REPORT_DIR'], task_name)
    os.makedirs(report_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    
    for model_name in base_cfg['MODELS']:
        print(f"\n--- Starting Model: {model_name} ({task_name}) ---")
        
        model_save_dir = os.path.join(base_cfg['MODEL_SAVE_DIR'], task_name)
        checkpoint_path = os.path.join(model_save_dir, f"{model_name}_best.pth")
        
        model = get_model(task_config['TYPE'], model_name, num_classes).to(device)
        
        start_time = time.time()
        
        if not eval_only:
            print("--- Starting Training ---")
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_cfg['LEARNING_RATE'])
            trained_model = train_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                device=device, 
                num_epochs=base_cfg['NUM_EPOCHS'],
                checkpoint_path=checkpoint_path
            )
        else:
            print(f"--- Skipping Training. Loading pre-trained model from {checkpoint_path} ---")
            if not os.path.exists(checkpoint_path):
                print(f"WARNING: Model file not found at {checkpoint_path}. Skipping evaluation.")
                results[model_name] = {}
                continue
            
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            trained_model = model

        end_time = time.time()
        computation_time = end_time - start_time
        
        print(f"\n--- Running Final Test Evaluation for {model_name} ---")
        model_report_name = f"{task_name}_{model_name}"
        test_metrics, _, _, _ = evaluate_classification_model(
            trained_model, test_loader, device, task_config['LABELS'], 
            report_dir, model_report_name
        )
        
        test_metrics['Computation Time (s)'] = computation_time if not eval_only else 0
        print(pd.Series(test_metrics).to_string(float_format="%.3f"))
        
        results[model_name] = test_metrics
        
    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'Model'})
    if not results_df.empty:
        chart_save_path = os.path.join(report_dir, f"{task_name}_f1_score_comparison.png")
        save_bar_chart(results_df, x_col='Model', y_col='F1-Score', 
                       title=f"{task_name} - Model F1-Score Comparison", 
                       save_path=chart_save_path)
                   
    return results

def run_study(tasks_to_run=None, eval_only=False):
    config = load_config()
    set_seed(config['BASE_CONFIG']['SEED'])
    
    device = torch.device(config['BASE_CONFIG']['DEVICE'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config['BASE_CONFIG']['MODEL_SAVE_DIR'], exist_ok=True)
    os.makedirs(config['BASE_CONFIG']['REPORT_DIR'], exist_ok=True)
    
    overall_results = {}
    
    tasks_to_execute = tasks_to_run if tasks_to_run is not None else config['TASKS'].keys()
    
    for task_name in tasks_to_execute:
        if task_name not in config['TASKS']:
            print(f"WARNING: Task '{task_name}' not found in config. Skipping.")
            continue
            
        task_config = config['TASKS'][task_name]
        task_config['BASE_CONFIG'] = {**config['BASE_CONFIG'], **task_config}
        task_config['BASE_CONFIG']['MODELS'] = config['MODELS']
        
        print(f"\n{'='*50}\n--- Running Task: {task_name} ({task_config['TYPE']}) ---\n{'='*50}")
        
        try:
            train_loader, val_loader, test_loader, num_classes = get_data_loaders(
                task_name, task_config['BASE_CONFIG']
            )
            
            if task_config['TYPE'] == 'classification':
                task_results = run_classification_task(
                    task_name, task_config, train_loader, val_loader, test_loader, device, eval_only=eval_only
                )
            elif task_config['TYPE'] == 'segmentation':
                task_results = run_segmentation_task(
                    task_name, task_config['BASE_CONFIG'], train_loader, val_loader, test_loader, eval_only=eval_only
                )
            
            for model_name, metrics in task_results.items():
                if metrics: 
                    overall_results[(task_name, model_name)] = metrics
                
        except Exception as e:
            print(f"ERROR: Skipping task {task_name} due to failure: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50 + "\n--- 📊 COMPARATIVE STUDY COMPLETE ---" + "\n" + "="*50)
    
    report_dir = config['BASE_CONFIG']['REPORT_DIR']

    dataset_df = create_dataset_details_table(config)
    print("\n--- Dataset Details ---")
    print(dataset_df.to_markdown(index=False))
    dataset_df.to_csv(os.path.join(report_dir, "report_dataset_details.csv"), index=False)
    
    model_comp_df = create_model_comparison_table(overall_results)
    if model_comp_df.empty:
        print("\nNo models were evaluated. Skipping final reports.")
    else:
        print("\n--- Model Comparison ---")
        print(model_comp_df.to_markdown(index=False, floatfmt=".3f"))
        model_comp_df.to_csv(os.path.join(report_dir, "report_model_comparison.csv"), index=False)
        
        best_model_df = create_best_model_table(model_comp_df)
        print("\n--- Best Model per Task ---")
        print(best_model_df.to_markdown(index=False, floatfmt=".3f"))
        best_model_df.to_csv(os.path.join(report_dir, "report_best_models.csv"), index=False)

        chart_save_path = os.path.join(report_dir, "report_best_models_comparison.png")
        best_model_df_copy = best_model_df.copy()
        best_model_df_copy['Score'] = best_model_df_copy.get('F1-Score', 0)
        if 'F1-Score' not in best_model_df_copy.columns: # Handle segmentation only
            best_model_df_copy['Score'] = best_model_df_copy.get('dice', 0)
        
        # --- FIX for FutureWarning ---
        save_bar_chart(best_model_df_copy, x_col='Task', y_col='Score', 
                       title="Best Model Performance per Task (F1/Dice)", 
                       save_path=chart_save_path)

    print(f"\n✅ All reports and plots saved to '{report_dir}' directory.")
    if not eval_only:
        print(f"✅ All models saved to '{config['BASE_CONFIG']['MODEL_SAVE_DIR']}' directory.")

if __name__ == "__main__":
    
    # --- Option 1: Run ALL tasks (Training + Evaluation) ---
    # run_study(eval_only=False)
    
    # --- Option 2: Run Evaluation ONLY (skips training) ---
    run_study(eval_only=True)
    
    # --- Option 3: Run Evaluation ONLY for a specific task ---
    # run_study(tasks_to_run=['M1_CT_Classification'], eval_only=True)
    # run_study(tasks_to_run=['M2_LIDC_Segmentation'], eval_only=True)
    # run_study(tasks_to_run=['M3_Rads_Classification'], eval_only=True)