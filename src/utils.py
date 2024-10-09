import torch
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

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