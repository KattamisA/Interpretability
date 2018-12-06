from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from IPython.core.debugger import set_trace

from skimage.measure import compare_psnr
from functions.utils.denoising_utils import *
from functions.models import *
from copy import deepcopy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor

imsize =-1
sigma = 25
sigma_ = sigma/255.

class global_values:
    
    net_input_saved = None
    noise = None
    out_avg = None
    last_net = None
    psnr_noisy_last = 0.0
    exp = None
    noise_std = 0.0
    PLOT = True
    img_np = None
    img_torch = None
    save = False
    net = None
    psnr_noisy = 0.0
    optimizer = None
    optimizer_last = None

def dip(img_np, arch = 'default', LR = 0.01, num_iter = 1000, exp_weight = 0.99, reg_noise_std = 1.0/30, INPUT = 'noise', save = False, save_path = '', plot = True, input_depth = 32):
    
    global_values.img_np = img_np.copy().astype(np.float32)
    global_values.img_np = global_values.img_np.transpose(2,0,1)/255.0
    global_values.img_torch = np_to_torch(global_values.img_np).type(dtype)
    
    pad = 'zero' 
    OPT_OVER = 'net' # 'net input'
    OPTIMIZER='adam' # 'LBFGS'
    
    ## Initialize global_value variables
    global_values.exp = exp_weight
    global_values.noise_std = reg_noise_std
    global_values.PLOT = plot
    global_values.psnr_noisy_last = 0.0
    global_values.out_avg = None
    global_values.save = save
    global_values.optimizer = None
    global_values.optimizer_last = None
    global_values.last_net = None

    if arch == 'default':
        #input_depth = 3
        net = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, act_fun='LeakyReLU').type(dtype)

    elif arch == 'complex':
        #input_depth = 32
        global_values.net = get_net(input_depth,'skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=4, 
                num_scales=5,
                upsample_mode='bilinear').type(dtype)

    elif arch == 'simple':
        input_depth = 3 
        net = get_net(input_depth,'skip', pad,
                skip_n33d=16, 
                skip_n33u=16, 
                skip_n11=0, 
                num_scales=3,
                upsample_mode='bilinear').type(dtype)
    else:
        assert False

    net_input = get_noise(input_depth, INPUT, (global_values.img_np.shape[1], global_values.img_np.shape[2])).type(dtype).detach()   
    global_values.net_input_saved = net_input.detach().clone()
    global_values.noise = net_input.detach().clone()
    
    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in global_values.net.parameters()]); 
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    
    if save == True:
        f= open("{}/Stats.txt".format(save_path),"w+")
        f.write("{:>11}{:>12}{:>12}\n".format('Iterations','Total_Loss','PSNR'))
    global_values.net.train()
    def closure(iter_value):
        show_every = 100
        figsize = 4
        
        ## Initialiaze/ Update variables
        if global_values.noise_std > 0.0:
            net_input = global_values.net_input_saved + (global_values.noise.normal_() * global_values.noise_std)

        out = global_values.net(net_input)

        ## Exponential Smoothing
        if global_values.out_avg is None:
            global_values.out_avg = out.detach()
        else:
            global_values.out_avg = global_values.out_avg * global_values.exp + out.detach() * (1 - global_values.exp)
        
        ## Calculate loss
        total_loss = mse(out, global_values.img_torch)
        total_loss.backward()

        global_values.psnr_noisy = compare_psnr(global_values.img_np, out.detach().cpu().numpy()[0]).astype(np.float32)
            
        print ('DIP Iteration {:>11}   Loss {:>11.7f}   PSNR_noisy: {:>5.4f}'
               .format(iter_value, total_loss.item(), global_values.psnr_noisy), end='\r')

        if global_values.PLOT == True and iter_value % show_every == 0:
            fig=plt.figure(figsize=(16, 16))
            fig.add_subplot(1, 3, 1)
            plt.imshow(np.clip(torch_to_np(out), 0, 1).transpose(1, 2, 0))
            plt.title('Output')
            fig.add_subplot(1, 3, 2)
            plt.imshow(np.clip(torch_to_np(global_values.out_avg), 0, 1).transpose(1, 2, 0))
            plt.title('Averaged Output')
            fig.add_subplot(1, 3, 3)
            plt.title('Original/Target')
            plt.imshow(global_values.img_np.transpose(1, 2, 0))
            plt.show()
            
        if  global_values.save and iter_value % show_every == 0:
            f = open("{}/Stats.txt".format(save_path),"a")
            f.write("{:>11}{:>12.8f}{:>12.8f}{:>12.8f}\n".format(
                iter_value, total_loss, global_values.psnr_noisy, global_values.psnr_noisy_last))
            plt.imsave("{}/it_{}.png".format(save_path,iter_value),
                       np.clip(torch_to_np(global_values.out_avg), 0, 1).transpose(1,2,0), format="png")

        # Backtracking   
        if (global_values.psnr_noisy_last - global_values.psnr_noisy) > 8.0:            
            print('\n Falling back to previous checkpoint.')
            global_values.net.load_state_dict(global_values.last_net.state_dict())
            global_values.optimizer.load_state_dict(global_values.optimizer_last.state_dict())            
            #for new_param, net_param in zip(global_values.last_net, net.parameters()):
                #net_param.detach().copy_(new_param)
            #return total_loss * 0
            global_values.save = False
            for j in range(iter_value % show_every - 1):
                global_values.optimizer.zero_grad()
                closure(iter_value - (iter_value % show_every) + j + 1)
                global_values.optimizer.step()                     
            global_values.optimizer.zero_grad()
            closure(iter_value)
            print('\n Return back to the original')                        
            global_values.save = True
            return total_loss           
            
        if (iter_value % show_every) == 0: 
            ## global_values.last_net = [x.detach().cuda() for x in net.parameters()]
            global_values.last_net = deepcopy(global_values.net)
            global_values.psnr_noisy_last = global_values.psnr_noisy
            global_values.optimizer_last = deepcopy(global_values.optimizer)
            
        return total_loss
        
    ### Optimize
    p = get_params(OPT_OVER, global_values.net, net_input)
    global_values.optimizer = torch.optim.Adam(p, lr=LR)    
    for j in range(num_iter):
        global_values.optimizer.zero_grad()
        closure(j)
        global_values.optimizer.step()
            
    print('\n')
    
    out = global_values.net(net_input)
    global_values.out_avg = global_values.out_avg * global_values.exp + out.detach() * (1 - global_values.exp)
    return global_values.out_avg
