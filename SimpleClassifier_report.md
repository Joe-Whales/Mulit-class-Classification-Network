# Classification Report for SimpleClassifier

## Model Architecture
```
SimpleClassifier(
  (fc1): Linear(in_features=5000, out_features=4, bias=True)
)
```

## K-fold Cross-validation Results
Mean Accuracy: 0.8953
Standard Deviation: 0.0001

![K-fold Cross-validation Accuracies](images/SimpleClassifier_kfold_accuracies.png)

## Training History
![Training History](images/SimpleClassifier_training_history.png)

## Test Set Results
Test Accuracy: 0.8964

## Classification Report
```
              precision    recall  f1-score   support

       world       0.86      0.86      0.86      1900
       sport       0.87      0.87      0.87      1900
    business       0.95      0.97      0.96      1900
    sci/tech       0.90      0.89      0.90      1900

    accuracy                           0.90      7600
   macro avg       0.90      0.90      0.90      7600
weighted avg       0.90      0.90      0.90      7600
```

## Confusion Matrix
![Confusion Matrix](images/SimpleClassifier_confusion_matrix.png)

## ROC Curve
![ROC Curve](images/SimpleClassifier_roc_curve.png)

## Model Summary
- Number of epochs: 10
- Batch size: 32
- Learning rate: 0.001
- Input size: 5000
- Number of classes: 4
