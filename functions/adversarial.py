from __future__ import print_function

import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from functions.utils.denoising_utils import *
from functions.saliency.saliency_utils import calculate_outputs_and_gradients, get_smoothed_gradients, pre_processing
from functions.saliency.visualization import linear_transform
from functions.classification import *

import matplotlib.pyplot as plt
import numpy as np
import cv2
from functions.utils.imagenet_classes import classes


def adversarial_examples(image_path, model_name='resnet18', method='Fast Gradient Sign Method', eps=5.0, alpha=1.0,
                         num_iter=None, show=True, cuda=False, image_name=''):
    
    if num_iter is None:
        num_iter = int(round(max(eps+4, eps*1.25)))
        
    if method == 'BI':
        method = 'Basic Iterative'
    elif method == 'LLCI':
        method = 'Least Likely Class Iterative'
    elif method == 'FGSM':
        method = 'Fast Gradient Sign Method'
    
    if show is True:
        print('Method: %s' % method)
        print('Model: %s \n' % model_name)

    model = getattr(models, model_name)(pretrained=True)

    orig = cv2.imread(image_path)[..., ::-1]
    
    # Reshape image to (3, 225, 256) and RGB (not BGR)
    # preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
    
    if model_name == 'inception_v3':
        orig = cv2.resize(orig, (300, 300))
    else:        
        orig = cv2.resize(orig, (224, 224))
    
    img = orig.copy().astype(np.float32)
    perturbation = np.empty_like(orig)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = (img - mean)/std
    img = img.transpose(2, 0, 1)

    #Set mode to evaluation and set criterion
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inp = Variable(torch.from_numpy(img).float().unsqueeze(0), requires_grad=True)

    # Predict before attack
    out = model(inp)
    sm = nn.Softmax(1)
    probs, ranks = sm(out).cpu().detach().sort(descending=True)
    pred = np.argmax(out.data.cpu().detach().numpy())

    if show:
        print('Prediction before attack:\n')
        print('{:<20}{:>20}\n'.format('Top 5 classes','Confidence'))
        for i in range(5):
            print('{:<20}{:>20.{prec}f}'.format(classes[int(ranks[0, i])].split(',')[0], probs[0, i], prec=5))
        print('\n')

    if method == 'Fast Gradient Sign Method':
        out = model(inp)
        loss = criterion(out, Variable(torch.Tensor([float(pred)]).long()))     # compute loss
        loss.backward()                                                         # compute gradients
        inp.data = inp.data + ((eps/255.0) * torch.sign(inp.grad.data))         # compute adversary

    if method == 'Basic Iterative' or method == 'Least Likely Class Iterative':
        
        orig_data = Variable(torch.from_numpy(img).float().unsqueeze(0))#, requires_grad=True)

        if method == 'Least Likely Class Iterative':
            y_target = ranks[0, -1]
        else:
            y_target = pred
        f = open("results/adversarial_examples/Adversarial_test/bi/{}.txt".format(image_name), "w+")
        for i in range(num_iter):
                out = model(inp)
                   
                # compute loss
                loss = criterion(out, Variable(torch.Tensor([float(y_target)]).long()))
                loss.backward()

                perturbation = (alpha/255.0) * torch.sign(inp.grad.data)    # calculate perturbation

                if method == 'Least Likely Class Iterative':
                    perturbation = - perturbation

                perturbation_sum = torch.clamp((inp.data + perturbation) - orig_data.data, min=-eps/255.0, max=eps/255.0)
                inp.data = orig_data.data + perturbation_sum
                inp.grad.data.zero_()

                ## Used for adversarial strength tests
                adv = inp.data.cpu().numpy()[0]
                adv = adv.transpose(1, 2, 0)
                adv = (adv * std) + mean
                adv = adv * 255.0
                # adv = adv[..., ::-1] # RGB to BGR
                adv = np.clip(adv, 0, 255).astype(np.uint8)
                confs, _ = classification(adv, sort=False, show=False, cuda=True)

                orig_conf = confs[0, pred]
                ll_conf = confs[0, ranks[0, -1]]

                f = open("results/adversarial_examples/Adversarial_test/bi/{}.txt".format(image_name), "a")
                f.write("{:>8} {:>15} {:>16.10f}\n".format(eps, orig_conf, ll_conf))

                if show is True:
                    pred_adv = np.argmax(model(inp).data.cpu().numpy())
                    sm = nn.Softmax(1)
                    confidence = sm(model(inp))
                    print("Iter [{:>3}/{:>3}]:  Prediction: {:<20}  Confidence: {:<10.3f}"
                          .format(i+1, num_iter, classes[pred_adv].split(',')[0], confidence[0, pred_adv]))

    if method == 'JSMA':
        if cuda:
            model.cuda()
            criterion.cuda()
            inp = inp.cuda()
        jsma_img = orig.astype(np.float32)
        y_target = ranks[0, -1]
        original_target = ranks[0, 0]

        # for i in range(num_iter):
        #     saliency_original = get_smoothed_gradients([jsma_img], model, original_target, calculate_outputs_and_gradients,
        #                                                     cuda=cuda, magnitude=False, stdev_spread=.01)
        #     saliency_original = -np.clip(saliency_original[0], -255, 0)  # only negative components
        #     for channel in range(3):
        #         saliency_original[:, :, channel] = linear_transform(saliency_original[:, :, channel], 99.9, 0, 0.0)
        #
        #     saliency_target = get_smoothed_gradients([jsma_img], model, y_target, calculate_outputs_and_gradients,
        #                                                     cuda=cuda, magnitude=False, stdev_spread=.01)
        #     saliency_target = np.clip(saliency_target[0], 0, 255)   # only positive components
        #     for channel in range(3):
        #         saliency_target[:, :, channel] = linear_transform(saliency_target[:, :, channel], 99.9, 0, 0.0)
        # f = open("results/adversarial_examples/jsma_simple/{}.txt".format(image_name), "w+")
        for i in range(num_iter):
            saliency_original, _ = calculate_outputs_and_gradients([jsma_img], model, original_target, cuda=cuda)
            saliency_original = -np.clip(saliency_original[0], -255, 0)  # only negative components
            saliency_original = np.transpose(saliency_original, (1, 2, 0))
            for channel in range(3):
                saliency_original[:, :, channel] = linear_transform(saliency_original[:, :, channel], 99.9, 0, 0.0)

            saliency_target, _ = calculate_outputs_and_gradients([jsma_img], model, y_target, cuda=cuda)
            saliency_target = np.clip(saliency_target[0], 0, 255)   # only positive components
            saliency_target = np.transpose(saliency_target, (1, 2, 0))

            for channel in range(3):
                saliency_target[:, :, channel] = linear_transform(saliency_target[:, :, channel], 99.9, 0, 0.0)

            adversarial_saliency = saliency_original * saliency_target

            perturbation = alpha * adversarial_saliency * 255.0 * 0.226
            perturbation_sum = np.clip((jsma_img + perturbation)-orig, 0, eps*0.226)
            jsma_img = perturbation_sum + orig

            inp = pre_processing(jsma_img, cuda=cuda)

            # Used for adversarial strength tests
            adv = inp.data.cpu().numpy()[0]
            adv = adv.transpose(1, 2, 0)
            adv = (adv * std) + mean
            adv = adv * 255.0
            # adv = adv[..., ::-1] # RGB to BGR
            adv = np.clip(adv, 0, 255).astype(np.uint8)
            confs, _ = classification(adv, sort=False, show=False, cuda=True)

            orig_conf = confs[0, original_target]
            ll_conf = confs[0, y_target]

            f = open("results/adversarial_examples/Adversarial_test/jsma_simple/{}.txt".format(image_name), "a")
            f.write("{:>8} {:>15} {:>16.10f}\n".format(eps, orig_conf, ll_conf))

            if show is True:
                pred_adv = np.argmax(model(inp).data.cpu().numpy())
                sm = nn.Softmax(1)
                Confidence = sm(model(inp))
                print("Iter [{:>3}/{:>3}]:  Prediction: {:<20}  Confidence: {:<10.3f}"
                      .format(i + 1, num_iter, classes[pred_adv].split(',')[0], Confidence[0, pred_adv]), end='\r')

    # predict on the adversarial image
    if show:
        sm = nn.Softmax(1)
        probs_adv, ranks_adv = sm(model(inp)).sort(descending=True)
        print('After attack: eps [%f] alpha [%f]\n' % (eps, alpha))
        print('{:<20}{:>20}\n'.format('Top 5 classes','Confidence'))
        for i in range(5):
            print('{:<20}{:>20.{prec}f}'.format(classes[int(ranks_adv[0, i])].split(',')[0], probs_adv[0, i], prec=5))

    # deprocess image
    adv = inp.data.cpu().numpy()[0]
    adv = adv.transpose(1, 2, 0)
    adv = (adv * std) + mean
    adv = adv * 255.0
    #adv = adv[..., ::-1] # RGB to BGR
    pert = adv - orig
    adv = np.clip(adv, 0, 255).astype(np.uint8)

    # if show == True:
    #     fig=plt.figure(figsize=(16, 16))
    #     fig.add_subplot(1, 3, 1)
    #     plt.subplots_adjust(wspace=0.5)
    #     plt.imshow(orig)
    #     plt.title('Original')
    #     fig.add_subplot(1, 3, 2)
    #     plt.imshow(np.sign(pert).astype(np.uint8))
    #     plt.title('Perturbation')
    #     fig.add_subplot(1, 3, 3)
    #     plt.title('Adversarial example')
    #     plt.imshow(adv)
    #     plt.show()
    #     fig.savefig('adv_example_{}.png'.format(classes[pred].split(',')[0]), bbox_inches='tight')

    return adv, orig, pert
