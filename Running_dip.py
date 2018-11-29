import matplotlib.pyplot as plt
import cv2
import argparse
import numpy as np
from utils import *
from functions.adversarial import *
from functions.dip import dip
from functions.classification import *
import torch


adv, orig, pert = adversarial_examples("data/goldfish.jpg",method = "LLCI",eps=100)

for i in range(7):
	std = 2^(i)/64
	out = dip(adv,num_iter=20000,save=True,PLOT=False,reg_noise_std = std,
				save_path='results/Std_complex_{}-64'.format(std),arch='complex',input_depth=32)
	
for i in range(5):
	id = 2^(i+1)
	out = dip(adv,num_iter=20000,save=True,PLOT=False,
				save_path='results/ID_complex_{}'.format(id),arch='complex',input_depth=id)
