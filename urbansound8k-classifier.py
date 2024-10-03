import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import soundata
import os

# Constants
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 * 4  # 4 seconds of audio
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

class UrbanSoundDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_data = self.data[idx]
        audio_path = clip_data.audio_path
        label = clip_data.tags[0]  # Assuming the first tag is the class label
        
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert stereo to mono

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        if waveform.shape[0] > NUM_SAMPLES:
            waveform = waveform[:NUM_SAMPLES]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, NUM_SAMPLES - waveform.shape[0]))

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec, self.data[idx].tags[0]

class CNNNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 10 * 10, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        train_loss = train_single_epoch(model, train_data_loader, loss_fn, optimiser, device)
        val_loss, val_accuracy = validate(model, val_data_loader, loss_fn, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    # Initialize and download the dataset
    dataset = soundata.initialize('urbansound8k')
    dataset.download()  # This will download the dataset if it's not already present

    # Load all clip data
    all_clips = list(dataset.clip_ids)
    clip_data = [dataset.clip(clip_id) for clip_id in all_clips]

    # Create dataset and split into train/validation
    full_dataset = UrbanSoundDataset(clip_data)
    train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)

    # Create data loaders
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # Get the number of unique classes
    num_classes = len(set(clip.tags[0] for clip in clip_data))

    # Instantiate the model and move it to the appropriate device
    cnn = CNNNetwork(num_classes).to(device)

    # Instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(cnn, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Save the model
    torch.save(cnn.state_dict(), "urbansound8k_model.pth")
    print("Model trained and stored at urbansound8k_model.pth")