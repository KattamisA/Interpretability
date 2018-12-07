import torch
import numpy as np
from functions.utils.common_utils import *
from functions.adversarial import *
from functions.dip import *
from generate_results import *


#adv, orig, pert = adversarial_examples("data/goldfish.jpg",method = "LLCI",eps=100, num_iter=125,show=False)

#for i in range(3):
#	std1 = 2**(i+3)
#        std = std1/64
#	out = dip(adv,num_iter=10000,save=True,PLOT=False,reg_noise_std = #std1,save_path='results/Adv_DIP/Std_complex_{}-64'.format(std1),arch='complex',input_depth=32)
#	
#for i in range(6):
#	ids = 2**(i+1)
#	out = dip(adv,num_iter=10000,save=True,PLOT=False,save_path='results/Adv_DIP/ID_complex_{}'.format(ids),arch='complex',input_depth=ids)

images = ['panda.jpg']#,'peacock.jpg','F16_GT.png', 'monkey.jpg','zebra_GT.png']
for i in images:
    save_path='results/Adv_DIP/{}'.format(i)
    num_iter = 101
    adv, orig, pert = adversarial_examples("data/{}".format(i), method = "LLCI", eps=1, show=False)
    out = dip(adv, num_iter=num_iter, save=True, plot=False, save_path=save_path, arch='complex')
    generate_result_files(save_path, adv, orig, num_iter)
    

