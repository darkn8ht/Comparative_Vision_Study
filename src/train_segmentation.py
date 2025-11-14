import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import segmentation_models_pytorch as smp
from .metrics import calculate_segmentation_metrics
from .losses import DiceLoss
from .reporting import save_bar_chart
import os
import numpy as np
import time
import pandas as pd

def get_segmentation_model(model_name):
    encoder_map = {
        'DenseNet-121': 'densenet121',
        'EfficientNet-B0': 'efficientnet-b0',
        'ResNet-50': 'resnet50',
        'MobileNet-V2': 'mobilenet_v2',
        'VGG-16': 'vgg16'
    }
    encoder_name = encoder_map.get(model_name)
    weights = "imagenet"
    
    return smp.Unet(
        encoder_name=encoder_name, 
        encoder_weights=weights,
        in_channels=1, 
        classes=1, 
        activation=None
    )

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="[TRAIN]"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def validate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="[VALID/TEST]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_segmentation_metrics(all_preds, all_targets)
    return epoch_loss, metrics

def run_segmentation_task(task_name, task_config, train_loader, val_loader, test_loader, eval_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Segmentation Training using device: {device}")
    
    MODELS = task_config.get('MODELS')
    num_epochs = task_config.get('NUM_EPOCHS', 10)
    learning_rate = task_config.get('LR', 1e-4)
    model_save_dir = os.path.join(task_config.get('MODEL_SAVE_DIR', 'models'), task_name)
    report_dir = os.path.join(task_config.get('REPORT_DIR', 'reports'), task_name)
    os.makedirs(report_dir, exist_ok=True)
    
    results = {}
    
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    criterion = lambda pred, target: bce_loss(pred, target) + dice_loss_fn(torch.sigmoid(pred), target)

    for model_name in MODELS:
        print(f"\n--- Starting Model: {model_name} ({task_name}) ---")
        
        model = get_segmentation_model(model_name).to(device)
        
        checkpoint_path = os.path.join(model_save_dir, f"{model_name}_best.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        start_time = time.time()
        
        if not eval_only:
            print("--- Starting Training ---")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_val_f1 = -1.0
            
            for epoch in range(1, num_epochs + 1):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_metrics = validate_model(model, val_loader, criterion, device)
                
                val_f1 = val_metrics['F1-Score']
                val_iou = val_metrics['iou']
                val_recall = val_metrics['Sensitivity']

                print(f"Epoch {epoch}/{num_epochs} | TRAIN Loss: {train_loss:.3f} | VAL Loss: {val_loss:.3f} | VAL F1: {val_f1:.3f} | VAL IoU: {val_iou:.3f} | VAL Recall: {val_recall:.3f}")
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"  --> Model checkpoint saved to {checkpoint_path}")
        
        else:
            print(f"--- Skipping Training. Loading pre-trained model from {checkpoint_path} ---")
            if not os.path.exists(checkpoint_path):
                print(f"WARNING: Model file not found at {checkpoint_path}. Skipping evaluation.")
                results[model_name] = {}
                continue

        end_time = time.time()
        computation_time = end_time - start_time

        print("\n--- Running Final Test Evaluation ---")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        _, test_metrics = validate_model(model, test_loader, criterion, device)
        
        test_metrics['Computation Time (s)'] = computation_time if not eval_only else 0
        
        # --- FIX: Use pd.Series to print all metrics ---
        print(pd.Series(test_metrics).to_string(float_format="%.3f"))
        results[model_name] = test_metrics
        
    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'Model'})
    if not results_df.empty:
        chart_save_path = os.path.join(report_dir, f"{task_name}_f1_score_comparison.png")
        save_bar_chart(results_df, x_col='Model', y_col='F1-Score', 
                       title=f"{task_name} - Model F1-Score Comparison", 
                       save_path=chart_save_path)

    return results