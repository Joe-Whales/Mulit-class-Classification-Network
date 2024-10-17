
# Training Report for cnn

## Training Progress

![Loss vs Epochs](docs/cnn_loss.png)

![Accuracy vs Epochs](docs/cnn_accuracy.png)

## Metrics Summary

### Training Metrics

| Metric | Initial | Final | Best |
|--------|---------|-------|------|
| Loss   | 1.9705 | 0.3130 | 0.2933 |
| Accuracy | 0.2715 | 0.8953 | 0.9121 |

### Validation Metrics

| Metric | Initial | Final | Best |
|--------|---------|-------|------|
| Loss   | 1.7487 | 0.7276 | 0.6759 |
| Accuracy | 0.3722 | 0.8061 | 0.8215 |

## Training Details

- Total epochs: 200
- Validation frequency: Every 5 epochs

## Confusion Matrix

![Confusion Matrix](docs/cnn_confusion_matrix.png)

Test Loss: 0.7811
Test Precision: 0.7977
Test F1 Score: 0.7800