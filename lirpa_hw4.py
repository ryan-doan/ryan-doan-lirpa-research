from hw4_handout import pytorch_ann, utils
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as functional
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

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
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=200, test_size=1, random_state=42)

# Flatten the images
X_train = X_train.reshape(X_train.shape[0], 784).float().to(device)
X_val = X_val.reshape(X_val.shape[0], 784).float().to(device)
X_test = X_test.reshape(X_test.shape[0], 784).float().to(device)

# Normalize the images
X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

# One-hot encode the labels
y_train = functional.one_hot(y_train).float().to(device)
y_val = functional.one_hot(y_val).float().to(device)
y_test = functional.one_hot(y_test).float().to(device)

model = pytorch_ann.PyTorchANN(input_size=784, output_size=10, hidden_size=32).to(device)
model.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=100,
    batch_size=64,
    learning_rate=0.2
)

x = torch.tensor([1., 1., 1., 1., 1.])
lower = torch.tensor([0, 0, 0, 0, 0])
upper = torch.tensor([1., 1., 1., 1., 1.])

lirpa_model = BoundedModule(model, torch.empty_like(x))
pred = lirpa_model(x)
print(f'Model prediction: {pred.item()}')