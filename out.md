Traceback (most recent call last):
  File "c:\Users\joewh\OneDrive\Desktop\Mulit-class-Classification-Network\test.py", line 196, in <module>
    main()
  File "c:\Users\joewh\OneDrive\Desktop\Mulit-class-Classification-Network\test.py", line 152, in main
    train_dataset = FruitDataset(root_dir=os.path.join(root_dir, 'train'), transform=SimpleTransform())
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\joewh\OneDrive\Desktop\Mulit-class-Classification-Network\test.py", line 21, in __init__
    self.classes = os.listdir(root_dir)
                   ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [WinError 3] The system cannot find the path specified: 'archive (6)/MY_data\\train'
(ai_assignment) PS C:\Users\joewh\OneDrive\Desktop\Mulit-class-Classification-Network> python -u "c:\Users\joewh\OneDrive\Desktop\Mulit-class-Classification-Network\test.py"  
Epoch 1/50
Train Loss: 2.0935, Train Acc: 20.34%
Test Loss: 1.8150, Test Acc: 29.66%
Epoch 2/50
Train Loss: 1.6995, Train Acc: 33.72%
Test Loss: 1.7074, Test Acc: 34.15%
Epoch 3/50
Train Loss: 1.6198, Train Acc: 38.59%
Test Loss: 1.7088, Test Acc: 33.95%
Epoch 4/50
Train Loss: 1.5633, Train Acc: 41.37%
Test Loss: 1.6484, Test Acc: 42.83%
Epoch 5/50
Train Loss: 1.5065, Train Acc: 45.20%
Test Loss: 1.6092, Test Acc: 43.41%
Epoch 6/50
Train Loss: 1.4639, Train Acc: 46.63%
Test Loss: 1.6016, Test Acc: 46.44%
Epoch 7/50
Train Loss: 1.4384, Train Acc: 48.76%
Test Loss: 1.5439, Test Acc: 45.95%
Epoch 8/50
Train Loss: 1.4233, Train Acc: 49.07%
Test Loss: 1.5195, Test Acc: 48.20%
Epoch 9/50
Train Loss: 1.4116, Train Acc: 49.50%
Test Loss: 1.5525, Test Acc: 46.34%
Epoch 10/50
Train Loss: 1.3753, Train Acc: 49.93%
Test Loss: 1.5200, Test Acc: 48.20%
Epoch 11/50
Train Loss: 1.3347, Train Acc: 53.24%
Test Loss: 1.4987, Test Acc: 52.10%
Epoch 12/50
Train Loss: 1.2990, Train Acc: 55.11%
Test Loss: 1.4614, Test Acc: 55.61%
Epoch 13/50
Train Loss: 1.2808, Train Acc: 56.11%
Test Loss: 1.5228, Test Acc: 48.39%
Epoch 14/50
Train Loss: 1.2558, Train Acc: 56.24%
Test Loss: 1.4132, Test Acc: 56.10%
Epoch 15/50
Train Loss: 1.2433, Train Acc: 57.58%
Test Loss: 1.3725, Test Acc: 55.61%
Epoch 16/50
Train Loss: 1.2056, Train Acc: 59.80%
Test Loss: 1.4648, Test Acc: 52.68%
Epoch 17/50
Train Loss: 1.1579, Train Acc: 60.89%
Test Loss: 1.3752, Test Acc: 59.02%
Epoch 18/50
Train Loss: 1.1827, Train Acc: 59.76%
Test Loss: 1.5461, Test Acc: 53.66%
Epoch 19/50
Train Loss: 1.1934, Train Acc: 59.97%
Test Loss: 1.4406, Test Acc: 57.37%
Epoch 20/50
Train Loss: 1.1465, Train Acc: 61.28%
Test Loss: 1.4631, Test Acc: 55.61%
Epoch 21/50
Train Loss: 1.1702, Train Acc: 60.67%
Test Loss: 1.4507, Test Acc: 57.95%
Epoch 22/50
Train Loss: 1.1174, Train Acc: 63.45%
Test Loss: 1.4142, Test Acc: 57.66%
Epoch 23/50
Train Loss: 1.1281, Train Acc: 62.41%
Test Loss: 1.3458, Test Acc: 57.27%
Epoch 24/50
Train Loss: 1.1222, Train Acc: 62.23%
Test Loss: 1.5169, Test Acc: 57.17%
Epoch 25/50
Train Loss: 1.0932, Train Acc: 63.67%
Test Loss: 1.4165, Test Acc: 56.88%
Epoch 26/50
Train Loss: 1.0730, Train Acc: 64.02%
Test Loss: 1.4849, Test Acc: 60.29%
Epoch 27/50
Train Loss: 1.0550, Train Acc: 65.06%
Test Loss: 1.3105, Test Acc: 59.90%
Epoch 28/50
Train Loss: 1.0629, Train Acc: 64.36%
Test Loss: 1.3464, Test Acc: 61.85%
Epoch 29/50
Train Loss: 1.0228, Train Acc: 66.01%
Test Loss: 1.3227, Test Acc: 63.22%
Epoch 30/50
Train Loss: 1.0250, Train Acc: 65.93%
Test Loss: 1.2484, Test Acc: 61.17%
Epoch 31/50
Train Loss: 1.0080, Train Acc: 66.06%
Test Loss: 1.2395, Test Acc: 61.07%
Epoch 32/50
Train Loss: 1.0094, Train Acc: 65.49%
Test Loss: 1.3191, Test Acc: 61.56%
Epoch 33/50
Train Loss: 0.9728, Train Acc: 68.06%
Test Loss: 1.2162, Test Acc: 64.39%
Epoch 34/50
Train Loss: 0.9720, Train Acc: 67.10%
Test Loss: 1.2302, Test Acc: 64.10%
Epoch 35/50
Train Loss: 0.9675, Train Acc: 67.41%
Test Loss: 1.2609, Test Acc: 64.20%
Epoch 36/50
Train Loss: 0.9901, Train Acc: 66.41%
Test Loss: 1.2338, Test Acc: 65.37%
Epoch 37/50
Train Loss: 0.9238, Train Acc: 69.06%
Test Loss: 1.2564, Test Acc: 61.85%
Epoch 38/50
Train Loss: 0.9138, Train Acc: 69.14%
Test Loss: 1.2044, Test Acc: 65.56%
Epoch 39/50
Train Loss: 0.9282, Train Acc: 69.36%
Test Loss: 1.2674, Test Acc: 64.39%
Epoch 40/50
Train Loss: 0.9066, Train Acc: 69.27%
Test Loss: 1.1748, Test Acc: 63.90%
Epoch 41/50
Train Loss: 0.8947, Train Acc: 71.23%
Test Loss: 1.2118, Test Acc: 65.27%
Epoch 42/50
Train Loss: 0.8841, Train Acc: 70.93%
Test Loss: 1.2406, Test Acc: 64.29%
Epoch 43/50
Train Loss: 0.8772, Train Acc: 70.75%
Test Loss: 1.2843, Test Acc: 64.78%
Epoch 44/50
Train Loss: 0.8725, Train Acc: 71.40%
Test Loss: 1.3801, Test Acc: 62.73%
Epoch 45/50
Train Loss: 0.8716, Train Acc: 71.66%
Test Loss: 1.3433, Test Acc: 62.83%
Epoch 46/50
Train Loss: 0.8586, Train Acc: 70.71%
Test Loss: 1.2026, Test Acc: 64.68%
Epoch 47/50
Train Loss: 0.8392, Train Acc: 72.58%
Test Loss: 1.2471, Test Acc: 66.54%
Epoch 48/50
Train Loss: 0.8270, Train Acc: 73.23%
Test Loss: 1.1373, Test Acc: 67.41%
Epoch 49/50
Train Loss: 0.8358, Train Acc: 71.10%
Test Loss: 1.1826, Test Acc: 66.44%
Epoch 50/50
Train Loss: 0.8077, Train Acc: 72.71%
Test Loss: 1.1974, Test Acc: 63.90%
Final Test Accuracy: 67.41%
Training complete. Best model saved as 'best_baseline_model.pth'.
Confusion matrix saved as 'baseline_confusion_matrix.png'.