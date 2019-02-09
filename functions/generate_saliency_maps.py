#import torch
import cv2
from torchvision import models
from functions.saliency.saliency_utils import calculate_outputs_and_gradients, generate_entrie_images, get_smoothed_gradients
from functions.saliency.integrated_gradients import *
from functions.saliency.visualization import visualize

import matplotlib.pyplot as plt
import numpy as np

def generate_saliency_maps(path, img_path, model_type='resnet18', cuda=False, top_percentile=99, bottom_percentile=1,
                           mask_mode=True, target_label_index = None):

    # start to create models...
    if model_type == 'inception_v3 ':
        model = models.inception_v3(pretrained=True)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif model_type == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        model_type = 'resnet18'
        model = models.resnet18(pretrained=True)

    model.eval()
    if cuda:
        model.cuda()
    # read the image
    #img = cv2.imread(path + '/' + img_path)
    img = cv2.imread('data/' + img_path)
    image_name = '{}'.format(img_path.split('.')[0])

    if model_type == 'inception_v3':
        img = cv2.resize(img, (300, 300))
    else:
        img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32)
    img = img[:, :, (2, 1, 0)]

    # calculate the gradient and the label index
    integration_steps = 50

    print('\nWorking on the GRAD saliency map')
    gradients, _ = calculate_outputs_and_gradients([img], model, target_label_index, cuda)
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=top_percentile,
                                     clip_below_percentile=bottom_percentile, overlay=True, mask_mode=mask_mode)
    img_gradient = visualize(gradients, np.empty_like(img), clip_above_percentile=top_percentile,
                             clip_below_percentile=bottom_percentile, overlay=False)

    print('Working on the SMOOTHGRAD saliency map')
    smoothedgrad_gradients = get_smoothed_gradients([img], model, target_label_index, calculate_outputs_and_gradients, cuda=True)
    smoothedgrad_gradients = smoothedgrad_gradients[0]
    img_smoothgrad_overlay = visualize(smoothedgrad_gradients, img, clip_above_percentile=top_percentile,
                                     clip_below_percentile=bottom_percentile, overlay=True, mask_mode=mask_mode)
    img_smoothgrad = visualize(smoothedgrad_gradients, np.empty_like(img), clip_above_percentile=top_percentile,
                             clip_below_percentile=bottom_percentile, overlay=False)

    # calculate the integrated gradients

    print('Working on the INTEGRATED GRAD saliency map')
    attributions = random_baseline_integrated_gradients(img, model, target_label_index, calculate_outputs_and_gradients,
                                                        steps=integration_steps, num_random_trials=10, cuda=cuda, smoothgrad=False)
    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=top_percentile,
                                                clip_below_percentile=bottom_percentile, overlay=True,
                                                mask_mode=mask_mode)
    img_integrated_gradient = visualize(attributions, np.empty_like(img), clip_above_percentile=top_percentile,
                                        clip_below_percentile=bottom_percentile, overlay=False)

    print('\nWorking on the INTEGRATED SMOOTHGRAD saliency map')
    smoothgrad_attributions = random_baseline_integrated_gradients(img, model, target_label_index, calculate_outputs_and_gradients,
                                                        steps=integration_steps, num_random_trials=10, cuda=cuda, smoothgrad=True)

    avg = np.average(smoothgrad_attributions, 2) * 255.0
    plt.imsave(path + '/Saliency_' + image_name + '_magnitude_true.png', np.uint8(avg), format="png")
    np.savetxt(path + '_magnitude_true_{}.txt', avg)

    img_integrated_smoothgrad_overlay = visualize(smoothgrad_attributions, img, clip_above_percentile=top_percentile,
                                                clip_below_percentile=bottom_percentile, overlay=True,
                                                mask_mode=mask_mode)
    img_integrated_smoothgrad = visualize(smoothgrad_attributions, np.empty_like(img), clip_above_percentile=top_percentile,
                                        clip_below_percentile=bottom_percentile, overlay=False)

    print('\nWorking on the INTEGRATED SMOOTHGRAD saliency map with magnitude = False')
    smoothgrad_attributions = random_baseline_integrated_gradients(img, model, target_label_index, calculate_outputs_and_gradients,
                             steps=integration_steps, num_random_trials=10, cuda=cuda, smoothgrad=True, magnitude=False)
    avg = np.average(smoothgrad_attributions, 2) * 255.0
    plt.imsave(path + '/Saliency_' + image_name + '_magnitude_false.png', np.uint8(avg), format="png")
    np.savetxt(path + '_magnitude_false_{}.txt', avg)
    img_integrated_smoothgrad_magn_overlay = visualize(smoothgrad_attributions, img, clip_above_percentile=top_percentile,
                                                clip_below_percentile=bottom_percentile, overlay=True,
                                                mask_mode=mask_mode)
    img_integrated_smoothgrad_magn = visualize(smoothgrad_attributions, np.empty_like(img), clip_above_percentile=top_percentile,
                                        clip_below_percentile=bottom_percentile, overlay=False)


    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_smoothgrad, img_smoothgrad_overlay,
                                        img_integrated_gradient, img_integrated_gradient_overlay, img_integrated_smoothgrad,
                                        img_integrated_smoothgrad_overlay, img_integrated_smoothgrad_magn, img_integrated_smoothgrad_magn_overlay)

    plt.imsave(path + '/Saliency_' + image_name + '_test.png', np.uint8(output_img), format="png")
    # plt.imsave(path + '/' + image_name + '.png', np.uint8(img_integrated_gradient), format="png")
    # plt.imshow(np.uint8(output_img))
    # plt.show()
    return
