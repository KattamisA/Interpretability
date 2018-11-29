from __future__ import print_function
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from DeepImagePrior.models import *

import torch
import torch.optim
import cv2

from skimage.measure import compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor

imsize =-1
sigma = 25
sigma_ = sigma/255.

def dip(img_np, arch = 'default', LR = 0.01, num_iter = 1000, exp_weight = 0.99, reg_noise_std = 1/30, INPUT = 'noise', save = False, save_path = '', PLOT = True,input_depth=32):
	img_np = img_np.transpose(2,0,1)
	img_torch = np_to_torch(img_np)
	pad = 'reflection' 
	OPT_OVER = 'net' # 'net,input'
	OPTIMIZER='adam' # 'LBFGS'
             
	show_every = 100
	figsize = 4

	if arch == 'default':
		input_depth = 3
		net = skip(
				input_depth, 3, 
				num_channels_down = [8, 16, 32, 64, 128], 
				num_channels_up   = [8, 16, 32, 64, 128],
				num_channels_skip = [0, 0, 0, 4, 4], 
				upsample_mode='bilinear',
				need_sigmoid=True, need_bias=True, act_fun='LeakyReLU').type(dtype)

	elif arch == 'complex':
		#input_depth = 32 
		net = get_net(input_depth,'skip', pad,
				skip_n33d=128, 
				skip_n33u=128, 
				skip_n11=4, 
				num_scales=5,
				upsample_mode='bilinear')
        
	elif arch == 'simple':
		input_depth = 32 
		net = get_net(input_depth,'skip', pad,
				skip_n33d=16, 
				skip_n33u=16, 
				skip_n11=4, 
				num_scales=3,
				upsample_mode='bilinear')

	else:
		assert False
    
	net_input = get_noise(input_depth, INPUT, (img_np.shape[1], img_np.shape[2])).type(dtype).detach()

	# Compute number of parameters
	s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
	print ('Number of params: %d' % s)

	# Loss
	mse = torch.nn.MSELoss().type(dtype)
	
	#####
	global i, out_avg, psrn_noisy_last, last_net, net_input_saved,net_input
	net_input_saved = net_input.detach().clone()
	noise = net_input.detach().clone()
	out_avg = None
	last_net = None
	psrn_noisy_last = 0
	reg_noise_std = 1/30
	i = 0
	if save == True:
		f= open("{}/Stats.txt".format(save_path),"w+")
		f.write("{:>5}{:>10}{:>10}\n".format('Iterations','Total Loss','PSNR'))
	
	def closure():
		
		global i, out_avg, psrn_noisy_last, last_net, net_input_saved,net_input
		
		if reg_noise_std > 0:
			net_input = net_input_saved + (noise.normal_() * reg_noise_std)
	   
		out = net(net_input)
	
		# Smoothing
		if out_avg is None:
			out_avg = out.detach()
		else:
			out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
			
		total_loss = mse(out, img_torch)
		total_loss.backward()
		
		psrn_noisy = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 

		# Note that we do not have GT for the "snail" example
		# So 'PSRN_gt', 'PSNR_gt_sm' make no sense
		print ('DIP Iteration {:>5}    Loss {:>7.6f}   PSNR_noisy: {:>5.4f}'
			   .format(i, total_loss.item(), psrn_noisy), end='\r')
		if  PLOT and i % show_every == 0:
			out_np = torch_to_np(out)
			fig=plt.figure(figsize=(16, 16))
			fig.add_subplot(1, 3, 1)
			plt.imshow(np.clip(out_np, 0, 1).transpose(1, 2, 0))
			plt.title('Output')
			fig.add_subplot(1, 3, 2)
			plt.imshow(np.clip(torch_to_np(out_avg), 0, 1).transpose(1, 2, 0))
			plt.title('Averaged Output')
			fig.add_subplot(1, 3, 3)
			plt.title('Original/Target')
			plt.imshow(img_np.transpose(1, 2, 0))
			window = 'DIP output'
			output = np.clip(torch_to_np(out_avg), 0, 1).transpose(1, 2, 0)

		if  save and i % show_every == 0:
			f = open("{}/Stats.txt".format(save_path),"a")
			f.write("{:>5}{:>12.8f}{:>12.8f}\n".format(i,total_loss,psrn_noisy))
			plt.imsave("{}/it_{}.png".format(save_path,i), np.clip(torch_to_np(out_avg), 0, 1).transpose(1,2,0), format="png")

		# Backtracking
		if i % show_every:
			if psrn_noisy - psrn_noisy_last < -5: 
				print('Falling back to previous checkpoint.')

				for new_param, net_param in zip(last_net, net.parameters()):
					net_param.detach().copy_(new_param)
	
				return total_loss*0
			else:
				last_net = [x.detach().cpu() for x in net.parameters()]
				psrn_noisy_last = psrn_noisy          
		i += 1
		
		return total_loss

	p = get_params(OPT_OVER, net, net_input)
	optimize(OPTIMIZER, p, closure, LR, num_iter)
	
	out = net(net_input)
	out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
	
	return out_avg
