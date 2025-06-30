import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim import optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms

from utils import get_loaders
from model import vit
import timeit
from tqdm import tqdm

train_dir_path = '../dataset/train.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
batch_size = 16
num_heads = 8
num_encoder = 12
img_size = 32
patch_size = 4
num_patches = (img_size // patch_size) ** 2
in_channels = 3
dropout = 0.1
embed_dim = in_channels * patch_size * patch_size
num_classes = 10
lr = 1e-3
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0


model = vit(in_channels, patch_size, embed_dim, num_patches, dropout, num_heads, num_encoder, num_classes,).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr,  weight_decay=WEIGHT_DECAY, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
scheduler = ExponentialLR(optimizer, gamma=0.5)
# train_dataloader, val_dataloader = get_loaders(train_dir_path, batch_size)
# train_dataset = torchvision.datasets.MNIST("C:\\Users\\wjxuan\\Documents\\PaperCode_implement\\mnist_data", train=True, download=True,
#                                      transform=torchvision.transforms.Compose(
#                                          [
#                                              torchvision.transforms.Resize(28),
#                                              torchvision.transforms.ToTensor(),
#                                              #  torchvision.transforms.Normalize([0.5], [0.5]),
#                                          ]
#                                                                              )
#                                      )
#
# val_dataset = torchvision.datasets.MNIST("C:\\Users\\wjxuan\\Documents\\PaperCode_implement\\mnist_data", train=False, download=True,
#                                      transform=torchvision.transforms.Compose(
#                                          [
#                                              torchvision.transforms.Resize(28),
#                                              torchvision.transforms.ToTensor(),
#                                              #  torchvision.transforms.Normalize([0.5], [0.5]),
#                                          ]
#                                                                              )
#                                      )
transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
train_data = torchvision.datasets.CIFAR10(root='../datasets', train=True, transform=transform, download=True)

test_data = torchvision.datasets.CIFAR10(root='../datasets', train=False, transform=transform, download=True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

start = timeit.default_timer()

for epoch in tqdm(range(epochs), position=0, leave=True):
    model.train()
    train_running_loss = 0
    train_label = []
    train_pred = []
    for idx, data in enumerate(tqdm(train_dataloader,position=0,leave=True)):
        img, label = data
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        pred_label = torch.argmax(pred, dim=1)

        train_label.extend(label.cpu().detach())
        train_pred.extend((pred_label.cpu().detach()))

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)
    scheduler.step()

    model.eval()
    val_loss = 0
    val_pred = []
    val_label = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_dataloader,position=0,leave=True)):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            pred_label = torch.argmax(pred, dim=1)

            val_label.extend(label.cpu().detach())
            val_pred.extend(pred_label.cpu().detach())

            loss = criterion(pred, label)
            val_loss += loss.item()
        val_loss = val_loss / (idx + 1)

    print('-'*30)
    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Train Accuracy:{sum(1 for x,y in zip(train_label, train_pred) if x == y) / len(train_label) :.4f}')
    print(f'Val Accuracy:{sum(1 for x,y in zip(val_pred, val_label) if x == y) / len(val_label) :.4f}')
    print('-'*30)


end = timeit.default_timer()
print(f'Time: {end - start:.2f}s')
