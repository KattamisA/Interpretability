from functions.adversarial import *
from functions.dip import *
from functions.generate_results import *
# from functions.adversarial import *
import cv2
from functions.classification import *

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']
q=0
for i in range(len(image_dataset)):
    image_path = image_dataset[i]
    image_name = '{}'.format(image_path.split('.')[0])
    print(image_name)
    orig = cv2.imread("data/" + image_path)[..., ::-1]
    _, ranks = classification(orig, sort=True, show=False, model_name='resnet18', cuda=True)
    orig_rank = ranks[0,0]
    adv = cv2.imread("results/adversarial_examples/Examples/FGSM_eps100/" + image_name + "FGSM_eps100.png")[..., ::-1]
    output = dip(adv, 'complex', 0.01, 300, save=False, plot=False, save_path=save_path, name=image_name)
    _, ranks = classification(output, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks[0,0] == orig_rank:
        q += 1
    print(q)