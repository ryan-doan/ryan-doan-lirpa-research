import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Define variables
CUDA = False
DATA_PATH = './data'
batch_size = 128
epochs = 5
lr = 2e-4
classes = 10
channels = 1
img_size = 64
latent_dim = 100
log_interval = 100
seed = 42

class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim):
        super(Generator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes)

        self.model = nn.Sequential(
            *self._create_layer(self.latent_dim + self.classes, 128, False),
            *self._create_layer(128, 256),
            *self._create_layer(256, 512),
            *self._create_layer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def _create_layer(self, size_in, size_out, normalize=True):
        layers = [nn.Linear(size_in, size_out)]
        if normalize:
            layers.append(nn.BatchNorm1d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise, labels):
        z = torch.cat((self.label_embedding(labels), noise), -1)
        x = self.model(z)
        x = x.view(x.size(0), *self.img_shape)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes)
        self.adv_loss = torch.nn.BCELoss()

        self.model = nn.Sequential(
            *self._create_layer(self.classes + int(np.prod(self.img_shape)), 1024, False, True),
            *self._create_layer(1024, 512, True, True),
            *self._create_layer(512, 256, True, True),
            *self._create_layer(256, 128, False, False),
            *self._create_layer(128, 1, False, False),
            nn.Sigmoid()
        )

    def _create_layer(self, size_in, size_out, drop_out=True, act_func=True):
        layers = [nn.Linear(size_in, size_out)]
        if drop_out:
            layers.append(nn.Dropout(0.4))
        if act_func:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, image, labels):
        x = torch.cat((image.view(image.size(0), -1), self.label_embedding(labels)), -1)
        return self.model(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)
    
class CGAN(nn.Module):
    def __init__(self, netG, netD, dataloader):
        super(CGAN, self).__init__()
        self.batch_size = 128
        self.epochs = 5
        self.lr = 2e-4
        self.classes = 10
        self.channels = 1
        self.latent_dim = 100
        self.log_interval = 100
        self.netG = netG
        self.netD = netD
        self.dataloader = dataloader

    def forward(self, noise, labels):
        return netG.forward(noise, labels)
    
    def save_weights(self):
        torch.save(self.netD.state_dict(), "cganD.pth")
        torch.save(self.netG.state_dict(), "cganG.pth")

    def load_weights(self):
        self.netD.load_state_dict(torch.load("C:\\Users\\doann\\Documents\\lirpa\\mnist_numbers\\cganD.pth"))
        self.netG.load_state_dict(torch.load("C:\\Users\\doann\\Documents\\lirpa\\mnist_numbers\\cganG.pth"))

    def train(self):
        # Setup Adam optimizers for both G and D
        optim_D = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optim_G = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))

        img_list = []

        netG.train()
        netD.train()
        #viz_z = torch.zeros((batch_size, latent_dim), device=device)
        viz_noise = torch.randn(self.batch_size, self.latent_dim, device=device)
        nrows = self.batch_size // 8
        viz_label = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(8)])).to(device)

        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(device), target.to(device)
                #batch_size = data.size(0)
                real_label = torch.full((data.shape[0], 1), 1., device=device)
                fake_label = torch.full((data.shape[0], 1), 0., device=device)

                # Train G
                netG.zero_grad()
                z_noise = torch.randn(self.batch_size, self.latent_dim, device=device)
                x_fake_labels = torch.randint(0, self.classes, (self.batch_size,), device=device)
                x_fake = netG(z_noise, x_fake_labels)
                y_fake_g = netD(x_fake, x_fake_labels)
                g_loss = netD.loss(y_fake_g, real_label)
                g_loss.backward()
                optim_G.step()

                # Train D
                netD.zero_grad()
                y_real = netD(data, target)
                d_real_loss = netD.loss(y_real, real_label)
                y_fake_d = netD(x_fake.detach(), x_fake_labels)
                d_fake_loss = netD.loss(y_fake_d, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optim_D.step()
                
                if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f}'.format(
                                epoch, batch_idx, len(self.dataloader),
                                d_loss.mean().item(),
                                g_loss.mean().item()))
                    
                    with torch.no_grad():
                        viz_sample = netG(viz_noise, viz_label)
                        img_list.append(vutils.make_grid(viz_sample, normalize=True))

if __name__ == '__main__':
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if CUDA:
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if CUDA else "cpu")
    cudnn.benchmark = True

    dataset = dset.MNIST(root=DATA_PATH, download=True,
                        transform=transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                        ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize generator and discriminator
    netG = Generator(classes, channels, img_size, latent_dim).to(device)
    print(netG)
    netD = Discriminator(classes, channels, img_size, latent_dim).to(device)
    print(netD)

    # Initialize CGAN
    cgan = CGAN(netG, netD, dataloader)

    # Train
    cgan.train()
    cgan.save_weights()

    # Plot the fake images
    noise = torch.randn(batch_size, latent_dim, device=device)
    label = torch.LongTensor(np.array([num for _ in range(16) for num in range(10)])).to(device)
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(cgan.forward(noise, label),(1,2,0)))
    plt.show()