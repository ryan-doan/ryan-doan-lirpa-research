from hw4_handout import pytorch_cnn, utils
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as functional
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import numpy as np
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU instead.")

def mnist_fashion_model(num_classes: int = 10):
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(num_features=32),
        nn.Dropout(p=0.3),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(num_features=64),
        nn.Dropout(p=0.3),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(num_features=128),
        nn.Dropout(p=0.3),

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.BatchNorm2d(num_features=256),
        nn.Dropout(p=0.3),

        nn.Flatten(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=num_classes)
    )
    return model

X_train, y_train, X_test, y_test = utils.load_mnist_f(return_tensor=True)

# Sample 2000 points for training and 200 points for testing
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=2000, test_size=200, random_state=42)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=2, test_size=1, random_state=42)

# Flatten the images
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).float().to(device)
X_val = X_val.reshape(X_val.shape[0], 1, 28, 28).float().to(device)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).float().to(device)

# Normalize the images
X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

# One-hot encode the labels
y_train = functional.one_hot(y_train).float().to(device)
y_val = functional.one_hot(y_val).float().to(device)
y_test = functional.one_hot(y_test).float().to(device)

state_dict = torch.load("C:\\Users\\doann\\Documents\\lirpa\\hw4_handout\\pytorch_cnn_weights.pth")

#model = mnist_fashion_model()
model = pytorch_cnn.PyTorchCNN(X_train, y_train, X_val, y_val, epochs=60, learning_rate=0.002, num_classes=10).to(device)
model.load_state_dict(torch.load("C:\\Users\\doann\\Documents\\lirpa\\hw4_handout\\pytorch_cnn_weights.pth"))
lirpa_model = BoundedModule(model, torch.empty_like(X_test), device=device)

eps = 1/255
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
X_test = BoundedTensor(X_test, ptb)

pred = lirpa_model(X_test)
pred2 = model.predict(X_test)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

print(pred2)
for i in range(0, 1):
    print(f'Image {i} top-1 prediction {label[i]} ground-truth {y_test[i].argmax()}')

lb, ub = lirpa_model.compute_bounds(x=(X_test,), method="CROWN")
for i in range(0, 1):
    for j in range(10):
        indicator = '(ground-truth)' if j == y_test[i][j] else ''
        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))