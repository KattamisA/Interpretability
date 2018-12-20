import torch
import numpy as np
from functions.utils.common_utils import *
from functions.dip import *
from generate_results import *


### Observing multiple images
images = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg']
for i in images:
    orig = cv2.imread(i)[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)
    std = 15
    img_noisy = img + std*np.random.randn(224,224,3)
    img_noisy = np.clip(img_noisy,0,255).astype(np.uint8)
    
    save_path='results/Denoising/Multiple_images/{}'.format(i.split('.')[0]) 
    out = dip(img_noisy, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='complex')
    generate_result_files(save_path, adv, orig, num_iter)
    
    save_path='results/Denoising/Multiple_images/EntropySGD/{}'.format(i.split('.')[0])
    out = dip(img_noisy, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='complex', OPTIMIZER = "EntropySGD", LR = 1)
    generate_result_files(save_path, adv, orig, num_iter)
    
    

