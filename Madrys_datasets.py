from functions.dip import *
# from functions.generate_results import *
from cifar_10.src.model.model import *
from cifar_10.src.utils.utils import *
from functions.utils.common_utils import np_to_torch

import matplotlib.pyplot as plt
import os
import torch
import torchvision as tv
import numpy as np
import cv2
from torch.utils.data import DataLoader



import matplotlib.pyplot as plt

model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

load_model(model, "checkpoint/cifar_10_default/checkpoint_12000.pth")

if torch.cuda.is_available():
    model.cuda()

data_path = "data/non_robust_CIFAR"

train_labels = torch.cat(torch.load(os.path.join(data_path, "CIFAR_lab")))
num_iter = 101
for i in range(10):
    print("############# Working on image: {}/500".format(i+1))
    image = cv2.imread(data_path + '/' + str(i) + '.png')[..., ::-1]
    save_path = 'results/Features/non_robust'
    output = dip(image, 'depth3', num_iter=num_iter, save=True, save_path=save_path, name=str(i))
    output = np.transpose(output,[2,0,1])
    confidences = model(np_to_torch(output))
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