from __future__ import print_function

import cv2
from functions.saliency import *

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'knife.jpg', 'rope.jpeg']

for i in range(len(image_dataset)):
    image = image_dataset[i]
    print('\n\n ### Working on image: ' + image.split('.')[0])
    generate_saliency_maps('results/Saliency', image, cuda=True, top_percentile=99, bottom_percentile=50, mask_mode=False)


