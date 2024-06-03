from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def load_data(train_size, test_size):
    # Download training data
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    # Download test data
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

    # Load data 
    loaders = {
        'train' : DataLoader(
            train_data, 
            batch_size=train_size, 
            shuffle=True, 
            num_workers=1
        ),
        
        'test'  : DataLoader(
            test_data, 
            batch_size=test_size, 
            shuffle=True, 
            num_workers=1
        )
    }
    return loaders