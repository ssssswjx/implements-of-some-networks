import os
import torch
import shutil
import numpy as np
from torch import nn
from model import (vit_base_patch16_224)


def set_seed(seed):
    """
    设置随机种子
    Args:
        seed: 随机种子

    Returns: None

    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(args):

    if args.model == "vit_base_patch16_224":
        model = vit_base_patch16_224(args.num_classes)
    else:
        raise Exception("Can't find any model name call {}".format(args.model))

    return model


def model_parallel(args, model):
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def remove_dir_and_create_dir(dir_name):
    """
    清除原有的文件夹，并且创建对应的文件目录
    Args:
        dir_name: 该文件夹的名字

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")