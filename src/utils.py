import torch
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import csv
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def log_metrics(filename, epoch, train_loss, train_acc, val_loss, val_acc):
    fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    is_new_file = not os.path.exists(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })