#import torch
import cv2
from torchvision import models
from functions.saliency.saliency_utils import calculate_outputs_and_gradients, generate_entrie_images, get_smoothed_gradients
from functions.saliency.integrated_gradients import *
from functions.saliency.visualization import visualize

import matplotlib.pyplot as plt
import numpy as np

def generate_saliency_maps(path, img_path, model_type='resnet18', cuda=False, top_percentile=99, bottom_percentile=1,
                           mask_mode=True, target_label=None, stdev_spread=.15, dual=False):

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
    img = cv2.imread(path + '/' + img_path)
    #img = cv2.imread('data/' + img_path)
    image_name = '{}'.format(img_path.split('.')[0])

    if model_type == 'inception_v3':
        img = cv2.resize(img, (300, 300))
    else:
        img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32)
    img = img[:, :, (2, 1, 0)]

    # calculate the gradient and the label index
    integration_steps = 30

    print('\nWorking on the GRAD saliency map')
    gradients, _ = calculate_outputs_and_gradients([img], model, target_label, cuda)
    gradients = np.transpose(gradients[0], (1, 2, 0))

    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=top_percentile,
                                     clip_below_percentile=0, overlay=True)
    img_gradient = visualize(gradients, img, clip_above_percentile=top_percentile,
                             clip_below_percentile=0, overlay=False)

    print('Working on the SMOOTHGRAD saliency map')
    smoothedgrad_gradients = get_smoothed_gradients([img], model, target_label, calculate_outputs_and_gradients,
                                                    cuda=True, magnitude=True, stdev_spread=stdev_spread)
    smoothedgrad_gradients = smoothedgrad_gradients[0]
    img_smoothgrad_overlay = visualize(smoothedgrad_gradients, img, clip_above_percentile=top_percentile,
                                     clip_below_percentile=0, overlay=True)
    img_smoothgrad = visualize(smoothedgrad_gradients, img, clip_above_percentile=top_percentile,
                             clip_below_percentile=0, overlay=False)

    # calculate the integrated gradients
    if dual is False:
        print('Working on the INTEGRATED GRAD saliency map')
        attributions = random_baseline_integrated_gradients(img, model, target_label, calculate_outputs_and_gradients,
                                                            steps=integration_steps, num_random_trials=10, cuda=cuda,
                                                            smoothgrad=False)
        img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=top_percentile,
                                                    clip_below_percentile=bottom_percentile, overlay=True)
        img_integrated_gradient = visualize(attributions, img, clip_above_percentile=top_percentile,
                                            clip_below_percentile=bottom_percentile, overlay=False)

        print('\nWorking on the INTEGRATED SMOOTHGRAD saliency map')
        smoothgrad_attributions = random_baseline_integrated_gradients(img, model, target_label, calculate_outputs_and_gradients,
                                                steps=integration_steps, num_random_trials=10, cuda=cuda, smoothgrad=True,
                                                                       stdev_spread=stdev_spread)

        img_integrated_smoothgrad_overlay = visualize(smoothgrad_attributions, img, clip_above_percentile=top_percentile,
                                                    clip_below_percentile=bottom_percentile, overlay=True)
        img_integrated_smoothgrad = visualize(smoothgrad_attributions, img, clip_above_percentile=top_percentile,
                                            clip_below_percentile=bottom_percentile, overlay=False)


        print('\nWorking on the INTEGRATED SMOOTHGRAD saliency map with magnitude = True')
        smoothgrad_attributions = random_baseline_integrated_gradients(img, model, target_label, calculate_outputs_and_gradients,
                                 steps=integration_steps, num_random_trials=10, cuda=cuda, smoothgrad=True, magnitude=True,
                                                                       stdev_spread=stdev_spread, absolute=True)

        img_integrated_smoothgrad_magn_overlay = visualize(smoothgrad_attributions, img, clip_above_percentile=top_percentile,
                                                    clip_below_percentile=20, overlay=True)
        img_integrated_smoothgrad_magn = visualize(smoothgrad_attributions, img, clip_above_percentile=top_percentile,
                                            clip_below_percentile=20, overlay=False)

        # Generating output image

        output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_smoothgrad, img_smoothgrad_overlay,
                                            img_integrated_gradient, img_integrated_gradient_overlay, img_integrated_smoothgrad,
                                            img_integrated_smoothgrad_overlay, img_integrated_smoothgrad_magn, img_integrated_smoothgrad_magn_overlay)

        #plt.imsave(path + '/Saliency_' + image_name + '.png', np.uint8(output_img), format="png")
        # if stdev_spread == 0.01:
        #     plt.imsave(path + '/Saliency_' + image_name + '_std.png', np.uint8(output_img), format="png")
        # else:
        plt.imsave(path + '/Saliency_' + image_name + '_test.png', np.uint8(output_img), format="png")

    else:
        print('\nWorking on the INTEGRATED SMOOTHGRAD saliency map with magnitude = True')
        smoothgrad_attributions = random_baseline_integrated_gradients(img, model, target_label,
                                                                       calculate_outputs_and_gradients,
                                                                       steps=integration_steps, num_random_trials=10,
                                                                       cuda=cuda, smoothgrad=True, magnitude=True,
                                                                       stdev_spread=stdev_spread, absolute=True)
        img_integrated_smoothgrad_magn_overlay = visualize(smoothgrad_attributions, img,
                                                           clip_above_percentile=top_percentile,
                                                           clip_below_percentile=20, overlay=True)
        img_integrated_smoothgrad_magn = visualize(smoothgrad_attributions, img, clip_above_percentile=top_percentile,
                                                   clip_below_percentile=20, overlay=False)

        print('\nWorking on the INTEGRATED SMOOTHGRAD saliency map with magnitude = True and absolute = False')

        smoothgrad_attributions_false = random_baseline_integrated_gradients(img, model, target_label,
                                                                       calculate_outputs_and_gradients,
                                                                       steps=integration_steps, num_random_trials=10,
                                                                       cuda=cuda, smoothgrad=True, magnitude=False,
                                                                       stdev_spread=stdev_spread, absolute=False)
        img_integrated_smoothgrad_false_overlay = visualize(smoothgrad_attributions_false, img,
                                                           clip_above_percentile=top_percentile,
                                                           clip_below_percentile=20, overlay=True)
        img_integrated_smoothgrad_false = visualize(smoothgrad_attributions_false, img, clip_above_percentile=top_percentile,
                                                   clip_below_percentile=20, overlay=False)

        img_integrated_smoothgrad_false_neg_overlay = visualize(smoothgrad_attributions_false, img,
                                                           clip_above_percentile=top_percentile,
                                                           clip_below_percentile=20, overlay=True, polarity='negative')
        img_integrated_smoothgrad_false_neg = visualize(smoothgrad_attributions_false, img, clip_above_percentile=top_percentile,
                                                   clip_below_percentile=20, overlay=False, polarity='negative')

        output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_smoothgrad, img_smoothgrad_overlay,
                                            img_integrated_smoothgrad_magn, img_integrated_smoothgrad_magn_overlay,
                                            img_integrated_smoothgrad_false, img_integrated_smoothgrad_false_overlay,
                                            img_integrated_smoothgrad_false_neg, img_integrated_smoothgrad_false_neg_overlay)

        plt.imsave(path + '/Saliency_' + image_name + '_test2.png', np.uint8(output_img), format="png")


    return
