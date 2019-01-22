import torch
import numpy as np
from functions.utils.common_utils import *
from functions.adversarial import *
from functions.dip import *
from generate_results import *

num_iter = 100
image_dataset = ['panda.jpg','peacock.jpg','F16_GT.png','monkey.jpg','zebra_GT.png','goldfish.jpg','whale.jpg', 'dolphin.jpg', 'spider.jpg','labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg', 'car_wheel.jpg','fountain.jpg', 'football_helmet.jpg','hourglass.jpg', 'refrigirator.jpg','knife.jpg','rope.jpeg']

for i in range(19,len(image_dataset)-1):
    for j in range(1,4):    
        image_path = image_dataset[i]
        image_name = '{}'.format(image_path.split('.')[0])
        save_path_common = 'results/Adv_DIP/Architecture_tests/{}'
                
        print("#############\n\nWorking on image: {}".format(image_name))           
        adv, orig, pert = adversarial_examples("data/{}".format(image_path), method = "LLCI", eps = 100, show=False)
      
        save_path=save_path_common.format('Adam/test{}'.format(j))
        out = dip(adv,'test{}'.format(j), 0.01, num_iter, save=True, plot=False, save_path = save_path)
        generate_result_files(save_path, adv, orig, num_iter, image_name)

        save_path=save_path_common.format('EntropySGD_std64/test{}'.format(j))
        out = dip(adv,'test{}'.format(j), 10, num_iter, 1/64., save=True, plot=False, save_path = save_path, OPTIMIZER = "EntropySGD")
        generate_result_files(save_path, adv, orig, num_iter, image_name)