import torch
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

def validate_model_classification(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            val_loss += criterion(output, target).item()
            predictions = torch.argmax(output, dim=1)
            y_true_list.extend(target.cpu().numpy())
            y_pred_list.extend(predictions.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
    return avg_val_loss, accuracy, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_path):
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)

        val_loss, acc, f1 = validate_model_classification(model, val_loader, criterion, device)

        print(f"  --> TRAIN Loss: {avg_train_loss:.3f} | VAL Loss: {val_loss:.3f} | ACC: {acc:.3f} | F1: {f1:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Model checkpoint saved to {checkpoint_path}")
            
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model