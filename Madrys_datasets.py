import os
import numpy as np
import torch as ch
from torchvision import transforms
import matplotlib.pyplot as plt

data_path = "data/non_robust_CIFAR"

train_data = ch.cat(ch.load(os.path.join(data_path, "CIFAR_ims")))
train_labels = ch.cat(ch.load(os.path.join(data_path, "CIFAR_lab")))

for i in range(500):
    image = train_data[i, :, :, :].numpy()
    image = np.transpose(image, [1, 2, 0])

    plt.imsave("data/non_robust_CIFAR/" + str(i) + ".png", np.uint8(image*255.0), format="png")
