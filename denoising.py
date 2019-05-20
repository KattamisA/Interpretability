from functions.dip import *
from functions.generate_results import *
import matplotlib.pyplot as plt


### Observing multiple images
num_iter = 5001
image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

for i in range(len(image_dataset)):
    image_path = image_dataset[i]
    image_name = '{}'.format(image_path.split('.')[0])
    orig = cv2.imread('data/{}'.format(image_path))[..., ::-1]
    img_noisy = cv2.imread('results/Denoising/dataset/{}_noisy.png'.format(image_name))[..., ::-1]
    save_path = 'results/Denoising/depth_tests/depth7'
    print('\n\n##### Working on image [{} , {}]'.format(i+1, image_name))
    _ = dip(img_noisy, 'depth7', 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
    generate_result_files(save_path, img_noisy, orig, num_iter, image_name)

# print('\n##################### LEARNING RATE INVESTIGATION')
# LR = [0.001, 0.1, 1]
# for i in range(18, len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     orig = cv2.imread('data/{}'.format(image_path))[..., ::-1]
#     img_noisy = cv2.imread('results/Denoising/dataset/{}_noisy.png'.format(image_name))[..., ::-1]
#     save_path_common = 'results/Denoising/lr_tests/{}'
#     print('\n##### Working on image [{} , {}]'.format(i+1, image_name))
#
#     for j in range(3):
#         print('LR = {}'.format(LR[j]))
#         save_path = save_path_common.format('lr{}'.format(j+1))
#         _ = dip(img_noisy, 'complex', LR[j], num_iter, save=True, plot=False, save_path=save_path, name=image_name)
#         generate_result_files(save_path, img_noisy, orig, num_iter, image_name)
#
# print('\n##################### KERNEL SIZE INVESTIGATION')
# for i in range(11, len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     orig = cv2.imread('data/{}'.format(image_path))[..., ::-1]
#     img_noisy = cv2.imread('results/Denoising/dataset/{}_noisy.png'.format(image_name))[..., ::-1]
#     save_path_common = 'results/Denoising/kernel_size_tests/{}'
#     print('\n##### Working on image [{} , {}]'.format(i + 1, image_name))
#     for j in range(3):
#         print('Kernel size = {}'.format(j+1))
#         save_path = save_path_common.format('kernel{}'.format(j+1))
#         _ = dip(img_noisy, 'kernel{}'.format(j+1), 0.01, num_iter, save=True, plot=False, save_path=save_path,
#                 name=image_name)
#         generate_result_files(save_path, img_noisy, orig, num_iter, image_name)
#
# print('\n##################### ITERATION NOISE INVESTIGATION')
# STD = [1./128.0, 1./64.0, 1./32.0, 1./16.0, 1./8.0, 1./4.0, 1./2.0]
# for i in range(len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     orig = cv2.imread('data/{}'.format(image_path))[..., ::-1]
#     img_noisy = cv2.imread('results/Denoising/dataset/{}_noisy.png'.format(image_name))[..., ::-1]
#     save_path_common = 'results/Denoising/std_investigation/{}'
#     print('\n##### Working on image [{} , {}]'.format(i + 1, image_name))
#     for j in range(7):
#         print('Iteration noise test: {}'.format(j+1))
#         save_path = save_path_common.format('std{}'.format(j+1))
#         _ = dip(img_noisy, 'complex', 0.01, num_iter, save=True, plot=False, save_path=save_path, reg_noise_std=STD[j],
#                 name=image_name)
#         generate_result_files(save_path, img_noisy, orig, num_iter, image_name)
#
# print('\n##################### SKIP CONNECTIONS INVESTIGATION')
# for i in range(len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     orig = cv2.imread('data/{}'.format(image_path))[..., ::-1]
#     img_noisy = cv2.imread('results/Denoising/dataset/{}_noisy.png'.format(image_name))[..., ::-1]
#     save_path_common = 'results/Denoising/skip_connections/{}'
#     print('\n##### Working on image [{} , {}]'.format(i + 1, image_name))
#     for j in range(6):
#         print('Skip_connections test: {}'.format(j+1))
#         save_path = save_path_common.format('skip{}'.format(j+1))
#         _ = dip(img_noisy, 'skip{}'.format(j+1), 0.01, num_iter, save=True, plot=False, save_path=save_path,
#                 name=image_name)
#         generate_result_files(save_path, img_noisy, orig, num_iter, image_name)
