from __future__ import print_function

import torch
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.device_count() else "cpu")
import torch.optim
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from skimage.measure import compare_psnr
from functions.utils.common_utils import *
from functions.models import *
from copy import deepcopy
from .global_parameters import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor

def dip(img_np, arch = 'default', LR = 0.01, num_iter = 1000, exp_weight = 0.99, reg_noise_std = 1.0/30, INPUT = 'noise', save = False, save_path = '', plot = True, input_depth = None, name = None, loss_fn = "MSE", OPTIMIZER = "adam", pad = 'zero',  OPT_OVER = 'net' ):
    
    glparam = global_parameters()
    glparam.set_params(save, plot, reg_noise_std, exp_weight)
    glparam.load_images(img_np)
    glparam.img_torch = glparam.img_torch.type(dtype)

    if arch == 'default':
        if input_depth == None:
            input_depth = 3
        net = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, act_fun='LeakyReLU').type(dtype)

    elif arch == 'complex':
        if input_depth == None:
            input_depth = 3
        glparam.net = get_net(input_depth,'skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=4, 
                num_scales=5,
                upsample_mode='bilinear').type(dtype)

    elif arch == 'simple':
        if input_depth == None:
            input_depth = 3 
        glparam.net = get_net(input_depth,'skip', pad,
                skip_n33d=16, 
                skip_n33u=16, 
                skip_n11=0, 
                num_scales=3,
                upsample_mode='bilinear').type(dtype)
    else:
        assert False
    
    
    
    net_input = get_noise(input_depth, INPUT, (glparam.img_np.shape[1], glparam.img_np.shape[2])).type(dtype).detach()   
    glparam.net_input_saved = net_input.detach().clone()
    glparam.noise = net_input.detach().clone()
    
    # Compute number of parameters
    param_numbers  = sum([np.prod(list(p.size())) for p in glparam.net.parameters()]); 
    print ('Number of params: %d' % param_numbers)

    # Loss function
    mse = torch.nn.MSELoss().type(dtype)
    
    if save == True:
        f= open("{}/Stats.txt".format(save_path),"w+")
        f.write("{:>11}{:>12}{:>12}\n".format('Iterations','Total_Loss','PSNR'))
        save_net_details(save_path, arch, param_numbers, pad, OPT_OVER, OPTIMIZER, input_depth,
                 loss_fn = 'Mean Squared Error', LR = LR, num_iter = num_iter, exp_weight = glparam.exp,
                 reg_noise_std = reg_noise_std, INPUT = 'INPUT', net = glparam.net)
                
    def closure(iter_value):
        show_every = 100
        figsize = 4
        
        ## Initialiaze/ Update variables
        if glparam.noise_std > 0.0:
            net_input = glparam.net_input_saved + (glparam.noise.normal_() * glparam.noise_std)
        out = glparam.net(net_input)

        ## Exponential Smoothing
        if glparam.out_avg is None:
            glparam.out_avg = out.detach()
        else:
            glparam.out_avg = glparam.out_avg * glparam.exp + out.detach() * (1 - glparam.exp)
        
        ## Calculate loss
        total_loss = mse(out, glparam.img_torch)
        total_loss.backward()

        glparam.psnr_noisy = compare_psnr(glparam.img_np, out.detach().cpu().numpy()[0]).astype(np.float32)
            
        print ('DIP Iteration {:>11}   Loss {:>11.7f}   PSNR_noisy: {:>5.4f}'.format(
            iter_value, total_loss.item(), glparam.psnr_noisy), end='\r')
        
        # Backtracking   
        if (glparam.psnr_noisy_last - glparam.psnr_noisy) > 7.0:            
            print('\n Falling back to previous checkpoint.')
            glparam.net.load_state_dict(glparam.last_net.state_dict())
            glparam.optimizer.load_state_dict(glparam.optimizer_last.state_dict())            
            for j in range(iter_value % show_every - 1):
                glparam.optimizer.zero_grad()
                closure(iter_value - (iter_value % show_every) + j + 1)
                glparam.optimizer.step()                     
            glparam.optimizer.zero_grad()
            closure(iter_value)
            print('\n Return back to the original')                        
            return total_loss           
            
        if (iter_value % show_every) == 0: 
            glparam.last_net = deepcopy(glparam.net)
            glparam.psnr_noisy_last = glparam.psnr_noisy
            glparam.optimizer_last = deepcopy(glparam.optimizer)
            
            if glparam.PLOT:
                fig=plt.figure(figsize=(16, 16))
                fig.add_subplot(1, 3, 1)
                plt.imshow(np.clip(torch_to_np(out), 0, 1).transpose(1, 2, 0))
                plt.title('Output')
                fig.add_subplot(1, 3, 2)
                plt.imshow(np.clip(torch_to_np(glparam.out_avg), 0, 1).transpose(1, 2, 0))
                plt.title('Averaged Output')
                fig.add_subplot(1, 3, 3)
                plt.title('Original/Target')
                plt.imshow(glparam.img_np.transpose(1, 2, 0))
                plt.show()
                
            if glparam.save:
                f = open("{}/Stats.txt".format(save_path),"a")
                f.write("{:>11}{:>12.8f}{:>12.8f}\n".format(iter_value, total_loss.item(), glparam.psnr_noisy))
                plt.imsave("{}/it_{}.png".format(save_path,iter_value),
                       np.clip(torch_to_np(glparam.out_avg), 0, 1).transpose(1,2,0), format="png")
                
            
        return total_loss
        
    ### Optimize
    glparam.net.train()
    p = get_params(OPT_OVER, glparam.net, net_input)
    glparam.optimizer = torch.optim.Adam(p, lr=LR)    
    for j in range(num_iter):
        glparam.optimizer.zero_grad()
        closure(j)
        glparam.optimizer.step()
            
    print('\n')
    
    out = glparam.net(net_input)
    glparam.out_avg = glparam.out_avg * glparam.exp + out.detach() * (1 - glparam.exp)
    return glparam.out_avg
