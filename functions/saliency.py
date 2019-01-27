import numpy as np
import torch
from torchvision import models
import cv2
import torch.nn.functional as F
from functions.saliency_utils import calculate_outputs_and_gradients, generate_entrie_images
from functions.integrated_gradients import random_baseline_integrated_gradients
from functions.visualization import visualize
import matplotlib.pyplot as plt


def generate_saliency_maps(path, img_path, model_type = 'resnet18', cuda = False):

    # start to create models...
    if model_type == 'inception_v3 ':
        model = models.inception_v3(pretrained=True)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif model_type == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        model_type = 'resnet18'
        print('Model not found, using resnet18 model instead')
        model = models.resnet18(pretrained=True)

    model.eval()
    if cuda:
        model.cuda()
    # read the image
    img = cv2.imread(path + '/' + img_path)

    if model_type == 'inception_v3':
        img = cv2.resize(img, (300, 300))
    else:
        img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32)
    #img = img[:, :, (2, 1, 0)]

    # calculate the gradient and the label index
    gradients, label_index = calculate_outputs_and_gradients([img], model, None, cuda)
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=1, overlay=True, mask_mode=True)
    img_gradient = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=1, overlay=False)

    # calculate the integrated gradients
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients,
                                                        steps=50, num_random_trials=10, cuda=cuda)
    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=1,
                                                overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=1, overlay=False)
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient,
                                        img_integrated_gradient_overlay)
    #cv2.imwrite(path + '/test', np.uint8(output_img))
    plt.imsave(path + '/test.png', np.uint8(output_img),format="png")
    plt.imshow(np.uint8(output_img))
    plt.show()
    return
