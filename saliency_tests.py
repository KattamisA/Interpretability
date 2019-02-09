import cv2
from functions.generate_saliency_maps import generate_saliency_maps
from functions.classification import classification
from functions.adversarial import *
import matplotlib.pyplot as plt

#image_dataset = ['labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
#                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
#                 'knife.jpg', 'rope.jpeg']
image_dataset = ['panda.jpg']

#'panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg','dolphin.jpg', 'spider.jpg',

#image_dataset = ['it_{}.png'.format(100*i) for i in range(0, 101)]
#image_dataset = ['it_{}.png'.format(100 * i) for i in range(57, 60)]

for i in range(len(image_dataset)):
    image = image_dataset[i]
    #orig = cv2.imread('data/knife.jpg')[..., ::-1]
    #_, ranks =classification(orig, model_name='resnet18', sort=True, show=False)
    #class_index = ranks[0, 0]

    print('###### Working on image: ' + image.split('.')[0])
    generate_saliency_maps('results/Saliency', image, model_type='resnet18', cuda=True,
                           top_percentile=99, bottom_percentile=0, mask_mode=True)

    #generate_saliency_maps('results/Adv_DIP/Multiple_images/knife', "adv_knife.png", model_type='resnet18', cuda=True,
    #                      top_percentile=99, bottom_percentile=0, mask_mode=True, target_label_index=class_index)
    print('\n')