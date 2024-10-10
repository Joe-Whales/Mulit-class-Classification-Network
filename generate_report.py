import os
import json
import matplotlib.pyplot as plt
from PIL import Image

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        epochs = []
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for line in lines:
            data = line.strip().split(',')
            epochs.append(int(data[0]))
            train_loss.append(float(data[1]))
            train_acc.append(float(data[2]))
            if len(data) > 3 and data[3] and data[4]:
                val_loss.append(float(data[3]))
                val_acc.append(float(data[4]))
            else:
                val_loss.append(None)
                val_acc.append(None)
    return epochs, train_loss, train_acc, val_loss, val_acc

def plot_metrics(epochs, train_metric, val_metric, metric_name, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metric, label=f'Train {metric_name}')
    
    # Plot validation metrics only for epochs where validation occurred
    val_epochs = [e for e, v in zip(epochs, val_metric) if v is not None]
    val_values = [v for v in val_metric if v is not None]
    plt.plot(val_epochs, val_values, label=f'Validation {metric_name}')
    
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Epochs')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def generate_report(metrics_file, confusion_matrix_file, output_folder):
    model_name = os.path.splitext(os.path.basename(metrics_file))[0].replace('_metrics', '')
    
    epochs, train_loss, train_acc, val_loss, val_acc = load_metrics(metrics_file)
    
    # Create docs folder if it doesn't exist
    os.makedirs(os.path.join(output_folder, 'docs'), exist_ok=True)
    
    # Generate plots
    plot_metrics(epochs, train_loss, val_loss, 'Loss', os.path.join(output_folder, 'docs', f'{model_name}_loss.png'))
    plot_metrics(epochs, train_acc, val_acc, 'Accuracy', os.path.join(output_folder, 'docs', f'{model_name}_accuracy.png'))
    
    # Copy confusion matrix to docs folder
    confusion_matrix_output = os.path.join(output_folder, 'docs', f'{model_name}_confusion_matrix.png')
    Image.open(confusion_matrix_file).save(confusion_matrix_output)
    
    # Filter out None values for validation metrics
    val_loss = [v for v in val_loss if v is not None]
    val_acc = [v for v in val_acc if v is not None]
    
    # Generate markdown report
    report = f"""
# Training Report for {model_name}

## Training Progress

![Loss vs Epochs](docs/{model_name}_loss.png)

![Accuracy vs Epochs](docs/{model_name}_accuracy.png)

## Metrics Summary

### Training Metrics

| Metric | Initial | Final | Best |
|--------|---------|-------|------|
| Loss   | {train_loss[0]:.4f} | {train_loss[-1]:.4f} | {min(train_loss):.4f} |
| Accuracy | {train_acc[0]:.4f} | {train_acc[-1]:.4f} | {max(train_acc):.4f} |

### Validation Metrics

| Metric | Initial | Final | Best |
|--------|---------|-------|------|
| Loss   | {val_loss[0]:.4f} | {val_loss[-1]:.4f} | {min(val_loss):.4f} |
| Accuracy | {val_acc[0]:.4f} | {val_acc[-1]:.4f} | {max(val_acc):.4f} |

## Training Details

- Total epochs: {len(epochs)}
- Validation frequency: Every 5 epochs

## Confusion Matrix

![Confusion Matrix](docs/{model_name}_confusion_matrix.png)

"""
    
    # Write the report
    with open(os.path.join(output_folder, f'{model_name}_report.md'), 'w') as f:
        f.write(report)

def main():
    results_folder = 'results'
    output_folder = 'reports'
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(results_folder):
        if file.endswith('_metrics.json'):
            metrics_file = os.path.join(results_folder, file)
            model_name = file.replace('_metrics.json', '')
            confusion_matrix_file = os.path.join(results_folder, f'{model_name}_confusion_matrix.png')
            
            if os.path.exists(confusion_matrix_file):
                generate_report(metrics_file, confusion_matrix_file, output_folder)
                print(f"Generated report for {model_name}")
            else:
                print(f"Confusion matrix not found for {model_name}")

if __name__ == "__main__":
    main()