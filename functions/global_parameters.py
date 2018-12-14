import torch
import numpy as np
from functions.utils.common_utils import *

class global_parameters:
    def __init__(self):           
        self.net_input_saved = None
        self.noise = None
        self.out_avg = None
        self.last_net = None
        self.psnr_noisy_last = 0.0
        self.exp = None
        self.noise_std = 0.0
        self.PLOT = True
        self.img_np = None
        self.img_torch = None
        self.save = False
        self.net = None
        self.psnr_noisy = 0.0
        self.optimizer = None
        self.optimizer_last = None
        self.interrupts = 0
        self.hardreset = 0
        
    def set_params(self, save, plot, reg_noise_std, exp_weight):
        self.save = save
        self.exp = exp_weight
        self.PLOT = plot
        self.noise_std = reg_noise_std
        
    def load_images(self, img_np):
        self.img_np = img_np.copy().astype(np.float32)
        self.img_np = self.img_np.transpose(2,0,1)/255.0
        self.img_torch = np_to_torch(self.img_np)