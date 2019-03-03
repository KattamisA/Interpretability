from __future__ import print_function

from functions.classification import *
import numpy as np
import cv2

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

methods = ['JSMA_eps1', 'JSMA_eps5', 'JSMA_eps25', 'JSMA_eps100']

for meth in methods:
    print("#######################\n\nMethod: {}".format(meth))
    for i in range(len(image_dataset)):
        orig_path = image_dataset[i]
        image_name = '{}'.format(orig_path.split('.')[0])

        orig = cv2.imread("data/{}".format(orig_path))[..., ::-1]
        _, ranks = classification(orig, sort=True, show=False, cuda=True)
        original_class = ranks[0, 0]

        image_path = '{}_{}.png'.format(image_name, meth)

        adversary = cv2.imread("results/adversarial_examples/Examples/{}/{}".format(meth, image_path))[..., ::-1]
        save_path_common = 'results/Adv_rev_noise/{}'
        save_path = save_path_common.format(meth)
        f = open("{}/{}_stats.txt".format(save_path, image_name), "w+")

        print("########\n\nWorking on image: {}".format(image_name))
        for std in range(1, 129):
            print('Noise standard deviation [{:>4}/128]'.format(std), end='\r')
            Average_confidence = 0
            for j in range(0, 5):
                adversary_noisy = adversary + std * np.random.randn(224, 224, 3)
                confidence, _ = classification(adversary_noisy, sort=False, show=False, cuda=True)
                Average_confidence += confidence[0, original_class]/5

            f = open("{}/{}_stats.txt".format(save_path, image_name), "a")
            f.write("{:>8}{:>16.10f}\n".format(std, Average_confidence))
