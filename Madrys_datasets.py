from functions.dip import *
from functions.generate_results import *
import matplotlib.pyplot as plt
import os
import torch
import torchvision as tv
import numpy as np

from torch.utils.data import DataLoader

from cifar_10.src.utils import makedirs, tensor2cuda, load_model, LabelDict
from cifar_10.src.model import WideResNet

import matplotlib.pyplot as plt

model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

load_model(model, "checkpoint/cifar-10_default/checkpoint_12000.pth")

if torch.cuda.is_available():
    model.cuda()

data_path = "data/non_robust_CIFAR"

train_labels = ch.cat(ch.load(os.path.join(data_path, "CIFAR_lab")))
num_iter = 101
for i in range(10):
    print("############# Working on image: {}/500".format(i+1))
    image = cv2.imread(data_path + '/' + str(i) + '.png')[..., ::-1]
    save_path = 'results/Features/non_robust'
    output = dip(image, 'depth3', num_iter=num_iter, save=True, save_path=save_path, name=str(i))
    confidences = model(output)
    print(confidences)

data_path = "data/robust_CIFAR"

train_labels = ch.cat(ch.load(os.path.join(data_path, "CIFAR_lab")))
num_iter = 101
for i in range(10):
    print("############# Working on image: {}/500".format(i+1))
    image = cv2.imread(data_path + '/' + str(i) + '.png')[..., ::-1]
    save_path = 'results/Features/robust'
    output = dip(image, 'depth3', num_iter=num_iter, save=True, save_path=save_path, name=str(i))
    # generate_result_files(save_path, output, image, num_iter, str(i), label=train_labels[i])