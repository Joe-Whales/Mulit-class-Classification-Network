import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .utils import get_device, calculate_metrics

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_y_true = []
    all_y_pred = []
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        all_y_true.extend(labels.cpu().numpy())
        all_y_pred.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_y_true, all_y_pred)
    metrics['loss'] = epoch_loss
    
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_y_true, all_y_pred)
    metrics['loss'] = epoch_loss
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, evaluate_every = 5, scheduler=None):
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")
        
        if (epoch + 1) % evaluate_every == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model = model.state_dict().copy()
        
        if scheduler:
            scheduler.step(train_metrics['loss'])
    
    return best_model

def get_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")