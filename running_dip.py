import torch
import numpy as np
from functions.utils.common_utils import *
from functions.adversarial import *
from functions.dip import *
from generate_results import *

#adv, orig, pert = adversarial_examples("data/goldfish.jpg", method = "LLCI",eps=100, show=False)
num_iter = 10001
#for i in range(6):
#    std1 = 2**(i)
#    std = std1/64.0
#    save_path = 'results/Adv_DIP/EntropySGD/Std_complex_{}-64'.format(std1)
#    out = dip(adv, num_iter=num_iter, save=True, plot=False, reg_noise_std = std1,
#              save_path = save_path, arch='complex', input_depth=32, OPTIMIZER = "EntropySGD", LR = 1)
#    generate_result_files(save_path, adv, orig, num_iter)

### Varying the learning rate for the entropy SGD method
#LRs = [20,50,100]
#for i in range(3):    
#    learning_rate = LRs[i]
#    print(learning_rate)
#    save_path = 'results/Adv_DIP/EntropySGD/LR_complex_{}'.format(i+8)
#    out = dip(adv, num_iter=num_iter, save=True, plot=False,
#              save_path = save_path, arch='complex', input_depth=32, OPTIMIZER = "EntropySGD", LR = learning_rate)
#    generate_result_files(save_path, adv, orig, num_iter)    

### Varying the input depth
#for i in range(6):
#    ids = 2**(i+1)
#    save_path = 'results/Adv_DIP/Goldfish/ID_complex_{}'.format(ids)    
#    out = dip(adv, num_iter=num_iter, save=True, plot=False,
#              save_path = save_path, arch='complex', input_depth=ids)
#    generate_result_files(save_path, adv, orig, num_iter)

### Observing multiple images #
images = ['panda.jpg','peacock.jpg','F16_GT.png','monkey.jpg','zebra_GT.png','goldfish.jpg','whale.jpg', 'dolphin.jpg', 'spider.jpg','labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg', 'car_wheel.jpg','fountain.jpg', 'football_helmet.jpg','hourglass.jpg', 'refrigirator.jpg','knife.jpg','rope.jpeg']

for i in image:
    adv, orig, pert = adversarial_examples("data/{}".format(i), method = "LLCI", eps = 100, show=False)
    print("#############\n\nWorking on image: {}".format(i.split('.')[0]))           
    name = '{}'.format(i.split('.')[0])
    
    save_path='results/Adv_DIP/Architecture/Adam'
    out = dip(adv, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='test')
    generate_result_files(save_path, adv, orig, num_iter, name)
    
    save_path='results/Adv_DIP/Architecture/EntropySGD'
    out = dip(adv, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='test', OPTIMIZER = "EntropySGD", LR = 10)
    generate_result_files(save_path, adv, orig, num_iter, name)
    
#input_depth = [64]
#for i in images:
#    for j in range(1):
#        adv, orig, pert = adversarial_examples("data/{}".format(i), method = "LLCI", eps = 100, show=False)
#        print("#############\n\n Input depth = {}  -  Working on image: {}".format(input_depth[j],i.split('.')[0]))       
#        name = '{}_ID{}'.format(i.split('.')[0],input_depth[j])
#        save_path='results/Adv_DIP/ID_investigation/Adam'.format(input_depth[j])
#        out = dip(adv, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='complex', input_depth = input_depth[j])
#        generate_result_files(save_path, adv, orig, num_iter, name)

        #save_path='results/Adv_DIP/ID_investigation/EntropySGD'
        #out = dip(adv, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='complex', OPTIMIZER = "EntropySGD", LR = 10, input_depth = input_depth[j])
        #generate_result_files(save_path, adv, orig, num_iter, name)    

        #save_path='results/Adv_DIP/Std_investigation/EntropySGD_LR10/{}'.format(i.split('.')[0])
        #out = dip(adv, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='complex', OPTIMIZER = "EntropySGD", LR = 10)
        #generate_result_files(save_path, adv, orig, num_iter)
