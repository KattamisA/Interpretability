import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
#from torchvision import transforms
from functions.utils.denoising_utils import *

import matplotlib.pyplot as plt
import numpy as np
import cv2
from functions.utils.imagenet_classes import classes


def adversarial_examples(image_path, model_name='resnet18', method='Fast Gradient Sign Method',eps = 5, alpha = 1,
                         num_iter=None, show=True, cuda=False):
    
    if num_iter == None:
        num_iter = int(round(max(eps+4,eps*1.25)))
        
    if method == 'BI':
        method = 'Basic Iterative'
    elif method == 'LLCI':
        method ='Least Likely Class Iterative'
    else:
        method = 'Fast Gradient Sign Method'
    
    if show == True:
        print('Method: %s' %(method))
        print('Model: %s \n' %(model_name))

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
    model.eval()#.cuda()
    criterion = nn.CrossEntropyLoss()#.cuda()

    # prediction before attack
    inp = Variable(torch.from_numpy(img).float().unsqueeze(0), requires_grad=True)#.cuda()
    if cuda:
        model.cuda()
        criterion.cuda()
        inp = inp.cuda()
    out = model(inp)
    sm = nn.Softmax(1)
    Probs, Ranks = sm(out).cpu().detach().sort(descending=True)

    pred = np.argmax(out.data.cpu().detach().numpy())

    if show == True:
        print('Prediction before attack:\n')
        print('{:<20}{:>20}\n'.format('Top 5 classes','Confidence'))
        for i in range(5):
            print('{:<20}{:>20.{prec}f}'.format(classes[int(Ranks[0,i])].split(',')[0], Probs[0,i],prec=5))
        print('\n')
    #inp = Variable(torch.from_numpy(img).float().unsqueeze(0), requires_grad=True)

    if method == 'Fast Gradient Sign Method':

        # compute loss
        out = model(inp)
        loss = criterion(out, Variable(torch.Tensor([float(pred)]).long()))

        # compute gradients
        loss.backward()

        # this is it, this is the method
        inp.data = inp.data + ((eps/255.0) * torch.sign(inp.grad.data))

        inp.grad.data.zero_() # this is just a compulsion, unnecessary here

    if method == 'Basic Iterative' or method == 'Least Likely Class Iterative':
        
        orig_data = Variable(torch.from_numpy(img).float().unsqueeze(0), requires_grad=True)
        sm = nn.Softmax(1)
        Probs_adv,Ranks_adv = sm(model(inp)).sort(descending=True)
        for i in range(num_iter):
                out = model(inp)

                if method == 'Least Likely Class Iterative':
                    y_target = Ranks_adv[0,-1]
                else:
                    y_target = pred
                   
                # compute loss
                loss = criterion(out, Variable(torch.Tensor([float(y_target)]).long()))
                loss.backward()

                # this is the method
                perturbation = (alpha/255.0) * torch.sign(inp.grad.data)

                if method == 'Least Likely Class Iterative':
                    perturbation = - perturbation

                perturbation_sum = torch.clamp((inp.data + perturbation) - orig_data.data, min=-eps/255.0, max=eps/255.0)
                inp.data = orig_data.data + perturbation_sum

                inp.grad.data.zero_() 

                # predict on the adversarial image, this inp is not the adversarial example we want, it's not yet clamped. And clamping can be done only after deprocessing.
                
                pred_adv = np.argmax(model(inp).data.cpu().numpy())
                sm = nn.Softmax(1)
                Confidence = sm(model(inp))
                #print("Iter [%3d/%3d]:  Prediction: %s  Confidence: %f"
                #      %(i+1, num_iter, classes[pred_adv].split(',')[0],Confidence[0,pred_adv]))
                if show == True:
                    print("Iter [{:>3}/{:>3}]:  Prediction: {:<20}  Confidence: {:<10.3f}"
                      .format(i+1, num_iter, classes[pred_adv].split(',')[0],Confidence[0,pred_adv]))

    # predict on the adversarial image
    sm = nn.Softmax(1)
    Probs_adv,Ranks_adv = sm(model(inp)).sort(descending=True)
    if show == True:
        print('After attack: eps [%f] alpha [%f]\n'
                    %(eps, alpha))
        print('{:<20}{:>20}\n'.format('Top 5 classes','Confidence'))
        for i in range(5):
            print('{:<20}{:>20.{prec}f}'.format(classes[int(Ranks_adv[0,i])].split(',')[0], Probs_adv[0,i],prec=5))

    # deprocess image
    adv = inp.data.cpu().numpy()[0]
    #perturbation = (adv-img).transpose(1,2,0)
    #perturbation = (perturbation * std) + mean
    #perturbation = perturbation * 255.0
    #v2.normalize((adv - img).transpose(1, 2, 0), perturbation, 0, 255, cv2.NORM_MINMAX, 0)
    adv = adv.transpose(1, 2, 0)
    adv = (adv * std) + mean
    adv = adv * 255.0
    #adv = adv[..., ::-1] # RGB to BGR
    pert = adv - orig
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    
    if show == True:
        fig=plt.figure(figsize=(16, 16))
        fig.add_subplot(1, 3, 1)
        plt.subplots_adjust(wspace=0.5)
        plt.imshow(orig)
        plt.title('Original')
        fig.add_subplot(1, 3, 2)
        plt.imshow(np.sign(pert).astype(np.uint8))
        plt.title('Perturbation')
        fig.add_subplot(1, 3, 3)
        plt.title('Adversarial example')
        plt.imshow(adv)
        plt.show()
        fig.savefig('adv_example_{}.png'.format(classes[pred].split(',')[0]), bbox_inches='tight')

    return adv, orig, pert