import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def read_metrics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        data = [line.strip().split(',') for line in lines]
    return {
        'epoch': [int(row[0]) for row in data],
        'train_loss': [float(row[1]) for row in data],
        'train_acc': [float(row[2]) for row in data],
        'val_loss': [float(row[3]) for row in data],
        'val_acc': [float(row[4]) for row in data]
    }

def plot_metric(folder_path, metric_name, title):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for file in json_files:
        file_path = os.path.join(folder_path, file)
        metrics = read_metrics(file_path)
        label = os.path.splitext(file)[0]
        
        plt.plot(metrics['epoch'], metrics[metric_name], label=f'{label}')
    
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(f'{title} Across Multiple Runs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'multi_run_comparison_{metric_name}.png'))
    plt.close()

def plot_all_metrics(folder_path):
    metrics_to_plot = [
        ('val_acc', 'Validation Accuracy'),
        ('val_loss', 'Validation Loss'),
        ('train_acc', 'Training Accuracy'),
        ('train_loss', 'Training Loss')
    ]
    
    for metric_name, title in metrics_to_plot:
        plot_metric(folder_path, metric_name, title)

# Usage
folder_path = 'results'  # Change this to your folder path
plot_all_metrics(folder_path)