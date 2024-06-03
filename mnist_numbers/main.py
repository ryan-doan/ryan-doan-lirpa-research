import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from load_data import load_data
from model import CNN

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train_size = 100
    test_size = 100
    loaders = load_data(train_size, test_size)

    # Initialize the model
    num_epochs = 10
    cnn = CNN(num_epochs, loaders)
    
    # Load pre-trained weights
    cnn.load_weights()

    # Load lirpa data
    test_data_size = 2
    lirpa_loaders = load_data(1, test_data_size)
    lirpa_batch = next(iter(lirpa_loaders['test']))
    test_data = lirpa_batch[0]
    test_label = lirpa_batch[1]

    # Initialize lirpa
    lirpa_model = BoundedModule(cnn, torch.empty_like(test_data))
    lirpa_model.eval()

    with torch.no_grad():
        eps = 1/255 # Perturb data by 1 pixel
        norm = float("inf")
        ptb = PerturbationLpNorm(norm=norm, eps = eps)
        test_data = BoundedTensor(test_data, ptb)

        pred = lirpa_model(test_data)
        label = torch.argmax(pred, dim=1).cpu().detach().numpy()

        for method in [
                'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
                print('Bounding method:', method)
                lb, ub = lirpa_model.compute_bounds(x=(test_data,), method=method.split()[0])
                for i in range(0, test_data_size):
                    print(f'Image {i} top-1 prediction {label[i]} ground-truth {test_label[i].argmax()}')
                    for j in range(10):
                        indicator = '(ground-truth)' if j == test_label[i] else ''
                        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))