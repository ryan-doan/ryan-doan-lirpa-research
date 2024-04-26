from hw4_handout import pytorch_cnn, utils
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as functional
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU instead.")

X_train, y_train, X_test, y_test = utils.load_mnist_f(return_tensor=True)

# Sample 2000 points for training and 200 points for testing
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=2000, test_size=200, random_state=42)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=1, test_size=1, random_state=42)

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

#Lower and upper bounds for perturbation.
lower = torch.tensor([[[0]*28]*28]).to(device)
upper = torch.tensor([[[255]*28]*28]).to(device)

ptb = PerturbationLpNorm(norm=np.inf, eps=10, x_L=0, x_U=255)
perturbedInput = BoundedTensor(X_test, ptb)
#print(f'X_test: {X_test[0][0]}')
#print(f'Perturbed X_test: {perturbedInput[0][0]}')
print(torch.eq(X_test, perturbedInput))