from functions.adversarial import *
from functions.dip import *
from functions.generate_results import *
import cv2

num_iter = 5001
image_dataset2 = [ 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

# 'panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 # 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 # 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg',
image_dataset = ['panda.jpg', 'monkey.jpg', 'goldfish.jpg', 'whale.jpg', 'knife.jpg']
        
# for i in range(4,5):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     save_path_common = 'results/Adv_DIP/Architecture_tests/{}'
#
#     print("#############\n\nWorking on image: {}".format(image_name))
#     orig = cv2.imread('data/'+image_path)[..., ::-1]
#     adv = cv2.imread("results/adversarial_examples/Examples/LLCI_eps100/" + image_name + "_LLCI_eps100.png")[..., ::-1]
#
#     for j in range(2,3):
#         print("####\n\nTest {}".format(j))
#         save_path = save_path_common.format('Adam/test{}'.format(j))
#         _ = dip(adv, 'test{}'.format(j), 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
#         generate_result_files(save_path, adv, orig, num_iter, image_name)
#
# for i in range(len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     save_path_common = 'results/Adv_DIP/Depth_tests/{}'
#
#     print("#############\n\nWorking on image: {}".format(image_name))
#     orig = cv2.imread('data/'+image_path)[..., ::-1]
#     adv = cv2.imread("results/adversarial_examples/Examples/LLCI_eps100/" + image_name + "_LLCI_eps100.png")[..., ::-1]
#
#     for j in range(2,3):
#         print("####\n\nDepth {}".format(j))
#         save_path = save_path_common.format('Adam/depth{}'.format(j))
#         _ = dip(adv, 'depth{}'.format(j), 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
#         generate_result_files(save_path, adv, orig, num_iter, image_name)
#
# for i in range(len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     save_path_common = 'results/Adv_DIP/Skip_connections/{}'
#
#     print("#############\n\nWorking on image: {}".format(image_name))
#     orig = cv2.imread('data/'+image_path)[..., ::-1]
#     adv = cv2.imread("results/adversarial_examples/Examples/LLCI_eps100/" + image_name + "_LLCI_eps100.png")[..., ::-1]
#
#     for j in range(2,3):
#         print("####\n\nSkip {}".format(j))
#         save_path = save_path_common.format('Adam/test{}'.format(j))
#         _ = dip(adv, 'skip{}'.format(j), 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
#         generate_result_files(save_path, adv, orig, num_iter, image_name)

for i in range(len(image_dataset2)):
    image_path = image_dataset2[i]
    image_name = '{}'.format(image_path.split('.')[0])
    save_path_common = 'results/Adv_DIP/Baselines/{}'

    print("#############\n\nWorking on image: {}".format(image_name))
    orig = cv2.imread('data/'+image_path)[..., ::-1]
    orig = cv2.resize(orig, (224,224))
    adv = cv2.imread("results/adversarial_examples/Examples/LLCI_eps100/" + image_name + "_LLCI_eps100.png")[..., ::-1]

    save_path1 = save_path_common.format('Original')
    _ = dip(orig, 'complex', 0.01, num_iter, save=True, plot=False, save_path=save_path1, name=image_name)
    generate_result_files(save_path1, orig, orig, num_iter, image_name)

    save_path2 = save_path_common.format('EntropySGD_lr1')
    _ = dip(adv, 'complex', 1, num_iter, save=True, plot=False, save_path=save_path2, name=image_name,  OPTIMIZER = "EntropySGD")
    generate_result_files(save_path2, adv, orig, num_iter, image_name)

    save_path3 = save_path_common.format('EntropySGD_lr10')
    _ = dip(adv, 'complex', 10, num_iter, save=True, plot=False, save_path=save_path3, name=image_name,  OPTIMIZER = "EntropySGD")
    generate_result_files(save_path3, adv, orig, num_iter, image_name)

# for i in range(len(image_dataset2)):
#     image_path = image_dataset2[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     save_path_common = 'results/Adv_DIP/Depth_tests/{}'
#
#     print("#############\n\nWorking on image: {}".format(image_name))
#     orig = cv2.imread('data/'+image_path)[..., ::-1]
#     adv = cv2.imread("results/adversarial_examples/Examples/LLCI_eps100/" + image_name + "_LLCI_eps100.png")[..., ::-1]
#
#     for j in range(7, 8):
#         print("####\n\nDepth {}".format(j))
#         save_path = save_path_common.format('Adam/depth{}'.format(j))
#         _ = dip(adv, 'depth{}'.format(j), 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
#         generate_result_files(save_path, adv, orig, num_iter, image_name)
#
# for i in range(len(image_dataset2)):
#     image_path = image_dataset2[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     save_path_common = 'results/Adv_DIP/Kernel_size_tests/{}'
#
#     print("#############\n\nWorking on image: {}".format(image_name))
#     orig = cv2.imread('data/'+image_path)[..., ::-1]
#     adv = cv2.imread("results/adversarial_examples/Examples/LLCI_eps100/" + image_name + "_LLCI_eps100.png")[..., ::-1]
#
#     for j in range(4):
#         print("####\n\nKernel 7x7")
#         save_path = save_path_common.format('Adam/kernel{}'.format(j))
#         _ = dip(adv, 'kernel{}'.format(j-1), 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
#         generate_result_files(save_path, adv, orig, num_iter, image_name)


# for i in range(len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     save_path_common = 'results/Adv_DIP/Std_investigation/{}'
#
#     print("#############\n\nWorking on image: {}".format(image_name))
#     adv, orig, pert = adversarial_examples("data/{}".format(image_path), method="LLCI", eps=100, show=False)
#
#     STD = [1/128., 1/64., 2/64., 8/64., 16/64., 32/64., 64/64.]
#     for j in range(7):
#         print("####\n\nTest {}".format(j+1))
#
#         save_path = save_path_common.format('EntropySGD/test{}'.format(j+1))
#         _ = dip(adv, 'complex', 10, num_iter, STD[j], save=True, plot=False, save_path=save_path, name=image_name, OPTIMIZER="EntropySGD")
#         generate_result_files(save_path, adv, orig, num_iter, image_name)