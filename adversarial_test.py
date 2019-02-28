from __future__ import print_function

from functions.adversarial import *
from functions.classification import *
import cv2

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

epsilons = [2*i + 1 for i in range(65)]

for i in range(len(image_dataset)):
    image_path = image_dataset[i]
    image_name = '{}'.format(image_path.split('.')[0])

    print('\n##### Working on image [{} , {}]'.format(i+1, image_name))

    orig = cv2.imread("data/{}".format(image_path))[..., ::-1]
    _, ranks = classification(orig, sort=True, show=False, cuda=True)
    original_class = ranks[0, 0]
    ll_class = ranks[0, -1]

    save_path = 'results/adversarial_examples'
    for eps in epsilons:
        print('Epsilon = [{:>4}/129]'.format(eps), end='\r')

        adv, _, _ = adversarial_examples("data/{}".format(image_path), method="FGSM", eps=eps, show=False, cuda=True)

        confs, _ = classification(adv, sort=False, show=False, cuda=True)

        orig_conf = confs[0, original_class]
        ll_conf = confs[0, ll_class]

        f = open("{}/{}_fgsm.txt".format(save_path, image_name), "a")
        f.write("{:>8} {:>8} {:>16.10f}\n".format(eps, orig_conf, ll_conf))


# plt.imsave("data/{}_JSMA.png".format(image_path.split('.')[0]), adv, format='png')


