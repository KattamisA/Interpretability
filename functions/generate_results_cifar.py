import cv2
import numpy as np
from cifar_10.src.model.model import *
from cifar_10.src.utils.utils import *
from functions.utils.common_utils import *
from skimage.measure import compare_psnr
import torch.nn as nn


def generate_result_files_cifar(path, orig, num_iter, name):
    sm = nn.Softmax()

    model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

    load_model(model, "cifar_model/cifar-10_default/checkpoint_12000.pth")
    num_images = int((num_iter-1)/10 + 1)

    Confidence = np.ones([num_images, 10])
    orig = orig.astype(np.float32)
    for i in range(num_images):
        loaded_image = cv2.imread('{}/it_{}.png'.format(path,i *10))[..., ::-1]
        img = loaded_image.copy().astype(np.float32)
        img = np.transpose(img, [2,0,1])
        img = np_to_torch(img)
        output_layer = model(img)
        Probs = sm(output_layer)
        Probs_np = torch_to_np(Probs)
        Confidence[i, :] = Probs_np

    np.savetxt('{}/{}_Confidences.txt'.format(path, name), Confidence)

    print('\nResults have been generated and stored in {}'.format(path))