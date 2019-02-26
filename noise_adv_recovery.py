from functions.classification import *
import numpy as np
import cv2

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

for i in range(len(image_dataset)):
    orig_path = image_dataset[i]
    orig = cv2.imread("data/{}".format(orig_path))[..., ::-1]
    _, ranks = classification(orig, sort=True, cuda=True)
    original_class = ranks[0, 0]

    image_path = '{}_LLCI_eps100.png'.format(orig_path.split('.')[0])
    image_name = '{}'.format(orig_path.split('.')[0])

    adversary = cv2.imread("results/adversarial_examples/{}".format(image_path))[..., ::-1]
    save_path_common = 'results/Adv_rev_noise/{}'
    save_path = save_path_common.format('LLCI_eps100')
    f = open("{}/{}_stats.txt".format(save_path, image_name), "w+")

    std = [i for i in range(1, 129)]
    print("#############\n\nWorking on image: {}".format(image_name))
    for s in range(128):
        Average_confidence = 0
        for j in range(0, 5):
            adversary_noisy = adversary + std[s] * np.random.randn(224, 224, 3)
            confidence, _ = classification(orig, sort=False, cuda=True)
            Average_confidence += confidence[0, original_class]/5

        f = open("{}/{}_stats.txt".format(save_path, image_name), "a")
        f.write("{:>8}{:>16.10f}\n".format(std[s], Average_confidence))
    print('Noise standard deviation {:>11}'.format(s), end='\r')
