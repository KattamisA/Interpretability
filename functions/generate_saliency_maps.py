#import torch
import cv2
from torchvision import models
from functions.saliency.saliency_utils import calculate_outputs_and_gradients, generate_entrie_images, get_smoothed_gradients
from functions.saliency.integrated_gradients import *
from functions.saliency.visualization import visualize

import matplotlib.pyplot as plt
import numpy as np

def generate_saliency_maps(path, img_path, model_type='resnet18', cuda=False, top_percentile=99, bottom_percentile=1, overlay=True, mask_mode=True):

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
    img = cv2.imread('data/' + img_path)
    image_name = '{}'.format(img_path.split('.')[0])

    if model_type == 'inception_v3':
        img = cv2.resize(img, (300, 300))
    else:
        img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32)
    img = img[:, :, (2, 1, 0)]

    # calculate the gradient and the label index
    gradients, label_index = calculate_outputs_and_gradients([img], model, None, cuda)
    gradients = np.transpose(gradients[0], (1, 2, 0))
    smoothedgrad_gradients = get_smoothed_gradients([img], model, label_index, calculate_outputs_and_gradients, cuda=True)
    smoothedgrad_gradients = smoothedgrad_gradients[0]
    img_gradient_overlay = visualize(smoothedgrad_gradients, img, clip_above_percentile=top_percentile, clip_below_percentile=bottom_percentile, overlay=overlay, mask_mode=mask_mode)
    img_gradient = visualize(smoothedgrad_gradients, img, clip_above_percentile=top_percentile, clip_below_percentile=bottom_percentile, overlay=False)
    print(np.shape(img_gradient))

    # calculate the integrated gradients
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients,
                                                        steps=50, num_random_trials=10, cuda=cuda, smoothgrad=True)
    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=top_percentile, clip_below_percentile=bottom_percentile,
                                                overlay=overlay, mask_mode=mask_mode)
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=top_percentile, clip_below_percentile=bottom_percentile, overlay=False)

    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient,
                                        img_integrated_gradient_overlay)
    plt.imsave(path + '/' + image_name + '.png', np.uint8(output_img), format="png")
    #plt.imsave(path + '/' + image_name + '.png', np.uint8(img_integrated_gradient), format="png")
    # plt.imshow(np.uint8(output_img))
    # plt.show()
    return
