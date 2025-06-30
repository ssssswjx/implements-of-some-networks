import torch
from torchvision.models import resnet50
from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count
from model import Vit

# Hyper Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5

BATCH_SIZE = 16
TRAIN_DF_DIR = "../dataset/train.csv"

# Model Parameters
IN_CHANNELS = 1
IMG_SIZE = 28
PATCH_SIZE = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
DROPOUT = 0.001

NUM_HEADS = 8
NUM_ENCODERS = 12
NUM_CLASSES = 10

LEARNING_RATE = 1e-4
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)

# model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT,
#             NUM_HEADS,  NUM_ENCODERS, NUM_CLASSES).to(device)

model = resnet50(pretrained=True)

tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)

flops = FlopCountAnalysis(model, tensor)

print('FLOPs:', flops.total())

print('Params table \n:', parameter_count_table(model))