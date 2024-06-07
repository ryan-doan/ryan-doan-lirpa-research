import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# Define the model
class CNN(nn.Module):
    def __init__(self, num_epochs, loaders):
        super(CNN, self).__init__()    
        self.num_epochs = num_epochs
        self.loaders = loaders    
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )        
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output   # return x for visualization
    
    def train_helper(self):
        loss_func = nn.CrossEntropyLoss()   
        optimizer = optim.Adam(self.parameters(), lr = 0.01) 
        self.train()
        
        # Train the model
        total_step = len(self.loaders['train'])
            
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.loaders['train']):
                
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(images)   # batch x
                b_y = Variable(labels)   # batch y
                output = self(b_x)[0]               
                loss = loss_func(output, b_y)
                
                # clear gradients for this training step   
                optimizer.zero_grad()           
                
                # backpropagation, compute gradients 
                loss.backward()                # apply gradients             
                optimizer.step()                
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item()))               
                    pass
            pass
        pass

    def eval_helper(self):
        self.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.loaders['test']:
                test_output, last_layer = self(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                pass
            print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
        pass

    def save_weights(self):
        torch.save(self.state_dict(), "mnist_weights.pth")

    def load_weights(self):
        self.load_state_dict(torch.load("C:\\Users\\doann\\Documents\\lirpa\\mnist_numbers\\mnist_weights.pth"))
        self.eval()