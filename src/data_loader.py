import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    images.append((img_path, self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config, train=True):
    if train and config.get('data_augmentation'):
        aug_config = config['data_augmentation']
        return transforms.Compose([
            transforms.RandomRotation(aug_config.get('random_rotation', 0)),
            transforms.RandomHorizontalFlip(p=1 if aug_config.get('random_horizontal_flip') else 0),
            transforms.RandomVerticalFlip(p=1 if aug_config.get('random_vertical_flip') else 0),
            transforms.ColorJitter(
                brightness=aug_config.get('color_jitter', {}).get('brightness', 0),
                contrast=aug_config.get('color_jitter', {}).get('contrast', 0),
                saturation=aug_config.get('color_jitter', {}).get('saturation', 0),
                hue=aug_config.get('color_jitter', {}).get('hue', 0)
            ),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

def create_data_loaders(config, data_dir, batch_size=32, num_workers=4):
    try:

        
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        #val_dir = os.path.join(data_dir, 'val')
        
        train_dataset = FruitDataset(train_dir, transform=get_transforms(config, train=True))
        test_dataset = FruitDataset(test_dir, transform=get_transforms(config, train=False))
        #val_dataset = FruitDataset(val_dir, transform=get_transforms(config, train=False))
        
        logger.info(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} test images")
        
        # Split test dataset into validation and test
        val_size = int(0.5 * len(test_dataset))
        test_size = len(test_dataset) - val_size
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created data loaders: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")
        
        return train_loader, val_loader, test_loader, train_dataset.classes
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise