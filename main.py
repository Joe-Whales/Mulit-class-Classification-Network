import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_loader import create_data_loaders
from src.models import get_model
from src.train import train_model, get_optimizer
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_learning_curves
from src.utils import set_seed, get_device
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Run fruit classification model')
    parser.add_argument('--config', type=str, default='config_simple.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Get device
    device = get_device()

    # Create data loaders
    train_loader, val_loader, test_loader, classes = create_data_loaders(
        config,
        config['data']['root_dir'],
        config['data']['batch_size'],
        config['data']['num_workers']
    )

    eval_mode = config.get('eval', False)
    
    # Create model
    model = get_model(config['model']['name'], config['model']['num_classes'], config).to(device)
    
    # Check if a checkpoint exists and load it
    if os.path.exists(config['paths']['best_model']):
        model.load_state_dict(torch.load(config['paths']['best_model']))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if not eval_mode:
        optimizer = get_optimizer(model, config['training']['optimizer'], config['training']['learning_rate'])

        # Train the model
        best_model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                                config['training']['num_epochs'], device, config, config['training']['evaluate_every'])

        # Save the best model
        torch.save(best_model, config['paths']['best_model'])

        # Load the best model for evaluation
        model.load_state_dict(best_model)

    # Evaluate the model
    test_metrics, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    
    # Save the predictions and true labels
    save_out_dir = "model_output"
    if not os.path.exists(save_out_dir):
        os.makedirs(save_out_dir)
    torch.save(y_true, os.path.join(save_out_dir, "y_true.pt"))
    torch.save(y_pred, os.path.join(save_out_dir, "y_pred.pt"))

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes, config['paths']['confusion_matrix'])

    # Plot learning curves (you'll need to modify the train_model function to return these)
    # plot_learning_curves(train_losses, val_losses, config['paths']['learning_curves'])

if __name__ == '__main__':
    main()