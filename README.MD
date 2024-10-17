# Fruit Classification Project

## Overview

This project implements a neural network-based system for fruit classification, addressing the challenge of accurately categorizing 9 distinct fruit classes using image data. We compare two architectures: a simple feedforward neural network as a baseline and an advanced Convolutional Neural Network (CNN).

## Table of Contents

- [Fruit Classification Project](#fruit-classification-project)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Models](#models)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Future Work](#future-work)
  - [Ethical Considerations](#ethical-considerations)
  - [Contributors](#contributors)

## Installation

To set up the project environment:

1. Clone the repository:
   ```
   git clone https://github.com/Joe-Whales/Mulit-class-Classification-Network.git
   cd fruit-classification
   ```

2. Create and activate a virtual environment:
   - On Windows:
     ```
     setup.bat
     ```
   - On Unix or MacOS:
     ```
     chmod +x setup.sh
     ./setup.sh
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The dataset is sourced from Kaggle and consists of images representing 9 different fruit classes: Apple, Orange, Kiwi, Mango, Pineapple, Strawberries, Banana, Cherry, and Watermelon. The data is split into training (70%), validation (15%), and test (15%) sets.

To use the dataset:
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class).
2. Extract the contents to a `data` folder in the project root.

## Project Structure

```
fruit-classification/
│
├── data/                  # Dataset folder (not included in repo)
├── src/
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── models.py          # Model architectures
│   ├── train.py           # Training loop and utilities
│   ├── evaluate.py        # Evaluation metrics and visualization
│   └── utils.py           # Utility functions
│
├── configs/               # Configuration files
│   ├── config_simple.yaml # Simple NN configuration
│   └── config_advanced.yaml # CNN configuration
│
├── main.py                # Main script to run training and evaluation
├── generate_configs.py    # Script to generate multiple configurations
├── generate_report.py     # Script to generate performance reports
├── graph_results.py       # Script to visualize results
│
├── setup.bat              # Windows setup script
├── setup.sh               # Unix/MacOS setup script
├── run_configs.bat        # Script to run multiple configurations
├── run_tests.bat          # Script to run tests
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Usage

To train and evaluate a model:

```
python main.py --config configs/config_advanced.yaml
```

To run multiple configurations:

```
run_configs.bat
```

To generate a report:

```
python generate_report.py
```

## Models

1. **Baseline: Simple Neural Network**
   - Feedforward neural network with three fully connected layers
   - Input: Flattened 224x224x3 RGB images
   - Hidden layers: 512 and 256 neurons with ReLU activation
   - Output: 10 neurons with softmax activation

2. **Advanced Model: Convolutional Neural Network (CNN)**
   - Three convolutional layers (64, 128, 256 filters) with ReLU activation
   - Max pooling after each convolutional layer
   - Adaptive average pooling
   - Fully connected layer for final classification
   - Incorporates data augmentation, batch normalization, and dropout

## Evaluation

We use the following metrics for evaluation:
- Precision
- Recall
- F1-score
- Confusion Matrix

Evaluation scripts can be found in `src/evaluate.py`.

## Results

- **Simple Neural Network**:
  - Test Loss: 8.1413
  - Test Precision: 0.4734
  - Test F1 Score: 0.4216

- **Convolutional Neural Network (CNN)**:
  - Test Loss: 0.7813
  - Test Precision: 0.7782
  - Test F1 Score: 0.7626

- **CNN Using Cleaned Dataset**:
  - Test Loss: 0.7811
  - Test Precision: 0.7977
  - Test F1 Score: 0.7800

## Future Work

- Expand and refine the dataset
- Explore advanced architectures (e.g., EfficientNet, Vision Transformers)
- Implement ensemble methods
- Utilize transfer learning from large-scale datasets

## Ethical Considerations

- Data collection and potential biases
- Environmental impact of model training
- Socioeconomic implications of automated fruit classification systems

## Contributors

- Aiden Harris (hrraid003@myuct.ac.za)
- Stuart Heath (hthstu002@myuct.ac.za)
- Joe Whales (whljos001@myuct.ac.za)

University of Cape Town, South Africa

Course Lecturer: Prof. Jan Buys

Dataset provided by Kaggle: https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class
