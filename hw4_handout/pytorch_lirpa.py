from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure
import random
import os
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def PyTorchCNN(num_classes: int = 10, ) -> None:
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


if __name__ == "__main__":
    import doctest
    import os

    from sklearn.model_selection import train_test_split
    from utils import (assert_greater_equal, load_mnist_f, print_blue,
                       print_green, print_red)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print_blue("GPU not available. Using CPU instead.")
        # device = torch.device("cpu")
        print_blue(f"Using device: {device}\n")

        X_train, y_train, X_test, y_test = load_mnist_f(return_tensor=True)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=2000, test_size=200, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=1, test_size=1, random_state=42)

        # Flatten the images (weight of size [32, 1, 3, 3], expected input[1, 2000, 28, 28])
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).float().to(device)
        X_val = X_val.reshape(X_val.shape[0], 1, 28, 28).float().to(device)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).float().to(device)

        # Normalize the images
        X_train = X_train / 255
        X_val = X_val / 255
        X_test = X_test / 255

        # One-hot encode the labels
        y_train = F.one_hot(y_train).float().to(device)
        y_val = F.one_hot(y_val).float().to(device)
        #y_test = F.one_hot(y_test).float().to(device)

        # Train the neural network
        pytorch_cnn = PyTorchCNN(num_classes=10).to(device)
        pytorch_cnn.load_state_dict(torch.load("C:\\Users\\doann\\Documents\\lirpa\\hw4_handout\\pytorch_cnn_weights.pth"))

        lirpa_model = BoundedModule(pytorch_cnn, torch.empty_like(X_test), device=device, bound_opts={'conv_mode': 'matrix'})
        pytorch_cnn.eval()
        lirpa_model.eval()

        with torch.no_grad():

            # Define bounds
            dim = X_test.shape
            lower = X_test.detach().clone()
            upper = X_test.detach().clone()
            eps = 1/255

            for i in range(dim[0]):
                for j in range(dim[2]):
                    for k in range(dim[3]):
                        if lower[i][0][j][k] >= eps:
                            lower[i][0][j][k] -= eps
                        if upper[i][0][j][k] <= 1 - eps:
                            upper[i][0][j][k] += eps

            norm = float("inf")
            ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
            X_test = BoundedTensor(X_test, ptb)
        
            pred = lirpa_model(X_test)
            label = torch.argmax(pred, dim=1).cpu().detach().numpy()

            for method in [
                'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
                print('Bounding method:', method)
                lb, ub = lirpa_model.compute_bounds(x=(X_test,), method=method.split()[0])
                for i in range(0, 1):
                    print(f'Image {i} top-1 prediction {label[i]} ground-truth {y_test[i].argmax()}')
                    for j in range(10):
                        indicator = '(ground-truth)' if j == y_test[i] else ''
                        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
    else:
        print_red("\nDoctests failed!\n")