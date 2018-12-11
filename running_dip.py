import torch
import numpy as np
from functions.utils.common_utils import *
from functions.adversarial import *
from functions.dip import *
from generate_results import *

adv, orig, pert = adversarial_examples("data/goldfish.jpg", method = "LLCI",eps=100, show=False)
num_iter = 10001
for i in range(7):
    std1 = 2**(i)
    std = std1/64.0
    save_path = 'results/Adv_DIP/goldfish/Std_complex_{}-64'.format(std1)
    out = dip(adv, num_iter=num_iter, save=True, plot=False, reg_noise_std = std1,
              save_path = save_path, arch='complex', input_depth=32)
    generate_result_files(save_path, adv, orig, num_iter)

#for i in range(7):
#    ids = 2**(i)
#    save_path = 'results/Adv_DIP/ID_complex_{}'.format(ids)    
#    out = dip(adv, num_iter=num_iter, save=True, PLOT=False,
#              save_path = save_path, arch='complex', input_depth=ids)
#    generate_result_files(save_path, adv, orig, num_iter)

#images = ['panda.jpg','peacock.jpg','F16_GT.png', 'monkey.jpg','zebra_GT.png', 'goldfish.jpg']
#for i in images:
#    save_path='results/Adv_DIP/{}'.format(i)
#    
#    adv, orig, pert = adversarial_examples("data/{}".format(i), method = "LLCI", eps=100, show=False)
#    out = dip(adv, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='complex')
#    generate_result_files(save_path, adv, orig, num_iter)
    

