from functions.adversarial import *
from functions.dip import *
from functions.generate_results import *

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

epsilon_values = [1, 5, 25, 100]
num_iter = 5001

for image in image_dataset:
    name = '{}'.format(image.split('.')[0])
    print("#############\n\nWorking on image: {} --- Method = LLCI".format(name))
    for epsilon in epsilon_values:
        img_path = 'results/adversarial_examples/Examples/JSMA_eps{}/{}_JSMA_eps{}.png'.format(epsilon, name, epsilon)
        adv = cv2.imread(img_path)[..., ::-1]
        orig = cv2.imread('data/' + image)[..., ::-1]
        save_path = 'results/Adv_DIP/All_adv_methods/JSMA_eps{}'.format(epsilon)
        _ = dip(adv, 'complex', 0.01, num_iter, save=True, save_path=save_path, plot=False, name=name)
        generate_result_files(save_path, adv, orig, num_iter, name, cuda=True, model='resnet18')


