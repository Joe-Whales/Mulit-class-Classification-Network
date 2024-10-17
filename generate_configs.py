import yaml
import itertools
import os

def generate_configs():
    # Base configuration
    base_config = {
        'data': {
            'root_dir': 'data',
            'num_workers': 6
        },
        'model': {
            'name': 'cnn',
            'num_classes': 9,
            'dropout_rate': 0.3
        },
        'training': {
            'num_epochs': 30,
            'optimizer': 'adam',
            'evaluate_every': 1,
            'scheduler': {
                'name': 'cosine_annealing',
                'min_eta': 0.0005,
                't_max': 20
            }
        },
        'data_augmentation': {
            'random_rotation': 15,
            'random_horizontal_flip': True,
            'random_vertical_flip': False,
            'color_jitter': {
                'brightness': 0.1,
                'contrast': 0.1,
                'saturation': 0.2,
                'hue': 0.1
            }
        },
        'seed': 42,
        'paths': {
            'best_model': 'results/cnn_best_model.pth',
            'confusion_matrix': 'results/cnn_confusion_matrix.png',
            'learning_curves': 'results/cnn_learning_curves.png',
            'log_dir': 'logs/cnn',
            'metrics': 'results/cnn_metrics.json'
        }
    }

    # Parameters to vary
    learning_rates = [0.01, 0.003]
    batch_sizes = [32, 64]
    weight_decays = [0.0001, 0.001]
    dropout_rates = [0.5, 0.3]

    # Generate all combinations
    combinations = list(itertools.product(learning_rates, batch_sizes, weight_decays, dropout_rates))

    # Ensure the configs directory exists
    os.makedirs('configs', exist_ok=True)

    # Generate config files
    for i, (lr, bs, wd, dr) in enumerate(combinations):
        config = base_config.copy()
        config['training']['learning_rate'] = lr
        config['data']['batch_size'] = bs
        config['training']['weight_decay'] = wd
        config['model']['dropout_rate'] = dr
        
        # Update paths for this specific configuration
        config['paths']['best_model'] = f'results/cnn_best_model_{i+1}.pth'
        config['paths']['confusion_matrix'] = f'results/cnn_confusion_matrix_{i+1}.png'
        config['paths']['learning_curves'] = f'results/cnn_learning_curves_{i+1}.png'
        config['paths']['log_dir'] = f'logs/cnn_{i+1}'
        config['paths']['metrics'] = f'results/cnn_metrics_{i+1}.json'
        
        filename = f'configs/config_{i+1}.yaml'
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Generated {filename}")

    return len(combinations)

if __name__ == "__main__":
    num_configs = generate_configs()
    print(f"Generated {num_configs} configuration files.")