from functions.adversarial import *
from functions.dip import *
from functions.generate_results import *

# image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
#                  'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
#                  'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
#                  'rope.jpeg', 'knife.jpg']
image_dataset = ['snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg', 'car_wheel.jpg', 'fountain.jpg',
                  'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg', 'rope.jpeg', 'knife.jpg']
epsilon_values = [1, 5, 25, 100]
num_iter = 5001

# for image in image_dataset:
#     name = '{}'.format(image.split('.')[0])
#     print("#############\n\nWorking on image: {} --- Method = FGSM".format(name))
#     for epsilon in epsilon_values:
#         adv, orig, _ = adversarial_examples("data/{}".format(image), method="FGSM", eps=epsilon, show=False, model_name='inception_v3')
#         save_path = 'results/Adv_DIP/Inception_v3/All_adv_methods/FGSM_eps{}'.format(epsilon)
#         _ = dip(adv, 'complex', 0.01, num_iter, save=True, save_path=save_path, plot=False, name=name)
#         generate_result_files(save_path, adv, orig, num_iter, name, cuda=True, model='inception_v3')


for image in ['refrigirator.jpg', 'rope.jpeg', 'knife.jpg']:
    name = '{}'.format(image.split('.')[0])
    print("#############\n\nWorking on image: {} --- Method = BI".format(name))
    for epsilon in epsilon_values:
        adv, orig, _ = adversarial_examples("data/{}".format(image), method="BI", eps=epsilon, show=False, model_name='inception_v3')
        save_path = 'results/Adv_DIP/Inception_v3/All_adv_methods/BI_eps{}'.format(epsilon)
        _ = dip(adv, 'complex', 0.01, num_iter, save=True, save_path=save_path, plot=False, name=name)
        generate_result_files(save_path, adv, orig, num_iter, name, cuda=True, model='inception_v3')

for image in image_dataset:
    name = '{}'.format(image.split('.')[0])
    print("#############\n\nWorking on image: {} --- Method = LLCI".format(name))
    for epsilon in epsilon_values:
        adv, orig, _ = adversarial_examples("data/{}".format(image), method="LLCI", eps=epsilon, show=False, model_name='inception_v3')
        save_path = 'results/Adv_DIP/Inception_v3/All_adv_methods/LLCI_eps{}'.format(epsilon)
        _ = dip(adv, 'complex', 0.01, num_iter, save=True, save_path=save_path, plot=False, name=name)
        generate_result_files(save_path, adv, orig, num_iter, name, cuda=True, model='inception_v3')

image_dataset2 = ['snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg', 'car_wheel.jpg', 'fountain.jpg',
                  'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg', 'rope.jpeg', 'knife.jpg']

# for image in image_dataset2:
#     name = '{}'.format(image.split('.')[0])
#     print("#############\n\nWorking on image: {} --- Method = FGSM".format(name))
#     for epsilon in epsilon_values:
#         adv, orig, _ = adversarial_examples("data/{}".format(image), method="FGSM", eps=epsilon, show=False, model_name='resnet18')
#         save_path = 'results/Adv_DIP/All_adv_methods/FGSM_eps{}'.format(epsilon)
#         _ = dip(adv, 'complex', 0.01, num_iter, save=True, save_path=save_path, plot=False, name=name)
#         generate_result_files(save_path, adv, orig, num_iter, name, cuda=True, model='resnet18')
#
# for image in image_dataset2:
#     name = '{}'.format(image.split('.')[0])
#     print("#############\n\nWorking on image: {} --- Method = BI".format(name))
#     for epsilon in epsilon_values:
#         adv, orig, _ = adversarial_examples("data/{}".format(image), method="BI", eps=epsilon, show=False, model_name='resnet18')
#         save_path = 'results/Adv_DIP/All_adv_methods/BI_eps{}'.format(epsilon)
#         _ = dip(adv, 'complex', 0.01, num_iter, save=True, save_path=save_path, plot=False, name=name)
#         generate_result_files(save_path, adv, orig, num_iter, name, cuda=True, model='resnet18')

for image in image_dataset:
    name = '{}'.format(image.split('.')[0])
    print("#############\n\nWorking on image: {} --- Method = LLCI".format(name))
    for epsilon in epsilon_values:
        adv, orig, _ = adversarial_examples("data/{}".format(image), method="LLCI", eps=epsilon, show=False, model_name='resnet18')
        save_path = 'results/Adv_DIP/All_adv_methods/LLCI_eps{}'.format(epsilon)
        _ = dip(adv, 'complex', 0.01, num_iter, save=True, save_path=save_path, plot=False, name=name)
        generate_result_files(save_path, adv, orig, num_iter, name, cuda=True, model='resnet18')


