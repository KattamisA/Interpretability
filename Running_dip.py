import matplotlib.pyplot as plt
import cv2
import argparse
import numpy as np
from utils import *
from functions.adversarial import *
from functions.dip import dip
from functions.classification import *
import torch


#adv, orig, pert = adversarial_examples("data/goldfish.jpg",method = "LLCI",eps=100, num_iter=125,show=False)

#for i in range(3):
#	std1 = 2**(i+3)
#        std = std1/64
#	out = dip(adv,num_iter=10000,save=True,PLOT=False,reg_noise_std = #std1,save_path='results/Adv_DIP/Std_complex_{}-64'.format(std1),arch='complex',input_depth=32)
#	
#for i in range(6):
#	ids = 2**(i+1)
#	out = dip(adv,num_iter=10000,save=True,PLOT=False,save_path='results/Adv_DIP/ID_complex_{}'.format(ids),arch='complex',input_depth=ids)

images = ['panda.jpg','peacock.jpg','F16_GT.png','monkey.jpg','zebra_GT.png']
for i in images:
    adv, orig, pert = adversarial_examples("data/{}".format(i),method = "LLCI",eps=100, num_iter=125,show=False)
    out = dip(adv,num_iter=10001,save=True,PLOT=False,save_path='results/Adv_DIP/{}'.format(i),arch='complex')
    