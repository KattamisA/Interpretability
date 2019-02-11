import torch
from torch.autograd import Variable
from torchvision import models

import numpy as np
import cv2
from functions.utils.imagenet_classes import classes

def classification(orig, model_name='resnet18', method='Fast Gradient Sign Method', sort = False, show = True, cuda=False):
    
    if show == True:
        print('Classification Model: %s \n' %(model_name))

    model = getattr(models, model_name)(pretrained=True)

    # Reshape image to (3, 224, 224) and RGB (not BGR)
    # preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
    
    if model_name == 'inception_v3':
        orig = cv2.resize(orig, (300, 300))
    else:        
        orig = cv2.resize(orig, (224, 224))
        
    img = orig.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = (img - mean)/std
    img = img.transpose(2, 0, 1)

    #Set mode to evaluation and set criterion
    model.eval()#.cuda()

    # prediction before attack
    inp = Variable(torch.from_numpy(img).float().unsqueeze(0))#.cuda()
    if cuda:
        inp= inp.cuda()
    out = model(inp)
    sm = torch.nn.Softmax(1)

    Probs,Ranks = sm(out).sort(descending=True).cpu().numpy()

    if show == True:    
        print('{:<20}{:>20}\n'.format('Top 5 classes', 'Confidence'))
        for i in range(5):
            print('{:<20}{:>20.{prec}f}'.format(classes[int(Ranks[0,i])].split(',')[0], Probs[0,i], prec=5))
        print('\n')
    
    if sort is False:
        sm = torch.nn.Softmax(1)
        Probs = sm(out)
        Ranks = []

    return Probs, Ranks
