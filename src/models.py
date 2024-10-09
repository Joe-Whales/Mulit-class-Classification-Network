import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(AdvancedCNN, self).__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32, dropout_rate),
            self._conv_block(32, 64, dropout_rate),
            self._conv_block(64, 128, dropout_rate),
            self._conv_block(128, 256, dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def _conv_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Update the get_model function
def get_model(model_name, num_classes, **kwargs):
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes)
    elif model_name == 'advanced_cnn':
        return AdvancedCNN(num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")