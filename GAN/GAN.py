import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

writer = SummaryWriter('logs')

save_dir = 'gan_images'
os.makedirs(save_dir, exist_ok=True)
num_epoch = 80
batch_size = 32
image_size = [1, 28, 28]
latent_dim = 96
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, np.prod(image_size)),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, z):
        # shape of z : [batch_size, latent_dim]
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)

        return image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod((image_size)), 1024),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )

    def forward(self, image):
        # shape of image : [batch_size, 1, 28, 28]
        prob = self.model(image.reshape(image.shape[0], -1))

        return prob


# Training
dataset = torchvision.datasets.MNIST('C:\\Users\\wjxuan\\Documents\\PaperCode_implement\\mnist_data', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         transforms.Resize(28),
                                         transforms.ToTensor(),
                                         # transforms.Normalize(mean=[0.5], std=[0.5])
                                     ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss = torch.nn.BCELoss().to(device)

total_i  = 0

for i in range(num_epoch):
    for idx, data in enumerate(dataloader):
        gt_images, _ = data
        gt_images = gt_images.to(device)

        z = torch.randn(batch_size, latent_dim).to(device)
        pred_images = generator(z)

        target = torch.ones(batch_size, 1).to(device)  # 设置为1，生成器优化令判别器判断不出

        g_loss = loss(discriminator(pred_images), target)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        target_real = torch.ones(batch_size, 1).to(device)
        target_f = torch.zeros(batch_size, 1).to(device)

        real_loss = loss(discriminator(gt_images), target_real)
        fake_loss = loss(discriminator(pred_images.detach()), target_f)
        # d_loss = 0.5*(loss(discriminator(gt_images), target)+ \
        #                   loss(discriminator(pred_images.detach()),  target_f)   )
        # 当real_loss和 fake _loss同时下降同时达到最小值，并且差不多大，说明D稳定
        total_i += 1
        # writer.add_scalar('real_loss', real_loss.item(), total_i)
        # writer.add_scalar('fake_loss', fake_loss.item(), total_i)

        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        if idx % 50 == 0:
            print(f'step:{len(dataloader)*i+idx}, d_loss: {d_loss.item():}, real_loss: {real_loss.item()}, fake_loss:{fake_loss.item()},g_loss: {g_loss.item():}')

        if idx % 1000  == 0:
            image = pred_images[:16].data.data.cpu()
            torchvision.utils.save_image(image, f'{save_dir}/image_{len(dataloader)*i+idx}.png', nrow=4)


