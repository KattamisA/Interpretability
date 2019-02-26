import matplotlib.pyplot as plt
import numpy as np

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

for i in range(len(image_dataset)):
    orig_path = image_dataset[i]
    image_path = '{}__LLCI_eps100.png'.format(orig_path.split('.')[0])
    image_name = '{}'.format(orig_path.split('.')[0])

    orig = cv2.imread("results/adversarial_examples/{}".format(image_name))[..., ::-1]
    save_path_common = 'results/Adv_rev_noise/{}'

    std = [i for i in range(1, 128)]

    for j in range(0,5):
        img_noisy = img + std * np.random.randn(224, 224, 3)
