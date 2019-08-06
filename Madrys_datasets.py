from functions.dip import *
from functions.generate_results_cifar import *

from cifar_10.src.model.model import *
from cifar_10.src.utils.utils import *

import os
import torch
import numpy as np
import cv2

model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

load_model(model, "checkpoint/cifar_10_default/checkpoint_12000.pth")

data_path = "data/non_robust_CIFAR"
save_path = 'results/Features/non_robust'

train_labels = torch.cat(torch.load(os.path.join(data_path, "CIFAR_lab")))
labels = train_labels.detach().numpy()
np.savetxt(save_path + "/labels.txt", labels)

num_iter = 1001
for i in range(10):
    print("############# Working on image: {}/500".format(i+1))
    image = cv2.imread(data_path + '/' + str(i) + '.png')[..., ::-1]
    _ = dip(image, 'depth3', num_iter=num_iter, save=True, save_path=save_path, name=str(i))
    generate_result_files_cifar(save_path, image, num_iter, str(i))


data_path = "data/robust_CIFAR"
save_path = 'results/Features/robust'

train_labels = torch.cat(torch.load(os.path.join(data_path, "CIFAR_lab")))
num_iter = 1001
for i in range(10):
    print("############# Working on image: {}/500".format(i+1))
    image = cv2.imread(data_path + '/' + str(i) + '.png')[..., ::-1]
    _ = dip(image, 'depth3', num_iter=num_iter, save=True, save_path=save_path, name=str(i))
    generate_result_files_cifar(save_path, image, num_iter, str(i))
    labels = train_labels.detach().numpy()[0]

np.savetxt(save_path + "/labels.txt", labels)
