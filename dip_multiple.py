from functions.adversarial import *
from functions.dip import *
from functions.generate_results import *

num_iter = 5001
image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']
        
for i in range(len(image_dataset)):
    image_path = image_dataset[i]
    image_name = '{}'.format(image_path.split('.')[0])
    save_path_common = 'results/Adv_DIP/Multiple_images'

    print("#############\n\nWorking on image: {}".format(image_name))
    adv, orig, pert = adversarial_examples("data/{}".format(image_path), method="LLCI", eps=100, show=False)

    # for j in range(4, 7):
    #     print("####\n\nTest {}".format(j))

    save_path = save_path_common#.format(''.format(j))
    _ = dip(adv, 'complex', 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
    generate_result_files(save_path, adv, orig, num_iter, image_name)

        # save_path = save_path_common.format('EntropySGD/test{}'.format(j+1))
        # _ = dip(adv, 'complex', 10, num_iter, STD[j], save=True, plot=False, save_path=save_path, name=image_name, OPTIMIZER="EntropySGD")
        # generate_result_files(save_path, advc, orig, num_iter, image_name)


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