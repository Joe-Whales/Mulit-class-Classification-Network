import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from .utils import get_device, calculate_metrics

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    metrics = calculate_metrics(all_y_true, all_y_pred)
    metrics['loss'] = test_loss
    
    return metrics, all_y_true, all_y_pred

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def plot_learning_curves(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.savefig(output_path)
    plt.close()