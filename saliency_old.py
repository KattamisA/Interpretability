import cv2
import numpy as np
import torch
from torchvision import models
from functions.classification import classification
from functions.utils.common_utils import *
from functions.integrated_gradients import *
from functions.Visualization_library import *

def model(img, target_class_index, model_name='resnet18'):
    criterion = torch.nn.CrossEntropyLoss() # .cuda()
    net = getattr(models, model_name)(pretrained=True)

    inp = np.asarray(img)
    predictions =np.ones([51,1000])
    grads = np.ones([51,224,224,3])
    net.eval()#.cuda()

    for i in range(inp.shape[0]):
        single_inp_init = inp[1,:,:,:]
        single_inp = single_inp_init.reshape(224,224,3)
        print('Iteration [{}/51]'.format(i+1), end='\r')
        p, _ = classification(single_inp, model_name, show=False)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        single_inp /= 255.0
        single_inp = (single_inp - mean) / std
        single_inp = single_inp.transpose(2, 0, 1)
        predictions[i,:] = torch_to_np(p)

        input = Variable(torch.from_numpy(single_inp).float().unsqueeze(0), requires_grad=True)

        output = net(input)
        loss = criterion(output, Variable(torch.Tensor([float(target_class_index)]).long()))
        loss.backward()

        grads_np = np.reshape(torch_to_np(input.grad.data), [3,224,224])
        grads_np = grads_np.transpose(1, 2, 0)
        grads_np = (grads_np * std) + mean
        grads_np = grads_np * 255.0
        grads[i,:,:,:] = grads_np

    return predictions, grads


def generate_saliency_maps(path, orig, num_iter, model_name='resnet18', method='Integrated Gradient'):

    p, r = classification(orig, model_name='resnet18', sort=True, show=False)
    original_class = r.detach().numpy()[0,0]
    original_class=original_class.astype(np.int16)
    #num_images = int((num_iter-1)/100 + 1)
    num_images = 1

    for i in range(num_images):
        #loaded_image = cv2.imread('{}/it_{}.png'.format(path,i*100))[..., ::-1]
        loaded_image = cv2.imread('data/peacock.jpg')[..., ::-1]
        if model_name == 'inception_v3':
            loaded_image = cv2.resize(loaded_image, (300, 300))
        else:        
            loaded_image = cv2.resize(loaded_image, (224, 224))
        img = loaded_image.copy().astype(np.float32)
        #img /= 255.0

        print("\n\nNew image")
        saliency_map, predictions = integrated_gradients(img, original_class, model)
        saliency_map2 = Visualize(saliency_map, img, clip_above_percentile=99, clip_below_percentile=1,
                                           overlay=True, plot_distribution=False)
        saliency_map = np.clip(saliency_map, 0, 255).astype(np.uint8)
        saliency_map2 = np.clip(saliency_map2, 0, 255).astype(np.uint8)
        plt.imsave("test1.png", saliency_map, format="png")
        plt.imsave("test2.png", saliency_map2, format="png")
        #plt.imsave("{}/sal_it_{}.png".format(path, i*100), saliency_map, format="png")

    print('Salency maps generated and stored in {}'.format(path))
