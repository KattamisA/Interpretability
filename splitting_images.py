import numpy as np
import cv2
import matplotlib.pyplot as plt

image_dataset2 = ['it_{}.png'.format(100*i) for i in range(0, 11)]
image_dataset2.extend(['it_{}.png'.format(200*i) for i in range(6, 51)])

# image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
#                  'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
#                  'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
#                  'rope.jpeg', 'knife.jpg']

path = "results/Saliency/adversarial/Knife_DIP/"
for i in range(len(image_dataset2)):
    im = image_dataset2[i]
    image_name = '{}'.format(im.split('.')[0])
    image_path = "Saliency_" + image_name + "_original_class.png.png"
    orig = cv2.imread(path + "{}".format(image_path))[..., ::-1]

    g = cv2.resize(orig[0:178, 187:366], (178, 178))
    ov_g = cv2.resize(orig[186:363, 187:366], (178, 178))

    sg = cv2.resize(orig[:, 375:553], (178, 364))
    ov_sg = cv2.resize(orig[186:363, 375:553], (178, 178))

    ig = cv2.resize(orig[0:178, 562:740], (178, 178))
    ov_ig = cv2.resize(orig[186:363, 562:740], (178, 178))

    igsg = cv2.resize(orig[0:178, 749:928], (178, 178))
    ov_igsg = cv2.resize(orig[186:363, 749:928], (178, 178))

    migsg = cv2.resize(orig[:, 936:1115], (178, 364))
    ov_migsg = cv2.resize(orig[186:363, 936:1115], (178, 178))

    # img_path = 'results/Adv_DIP/Multiple_images/knife/' + name + '.png'

    output = cv2.imread('results/Adv_DIP/Multiple_images/knife/' + image_name + '.png')[..., ::-1]
    output = cv2.resize(output, (178,178))
    blank_hor = np.ones((8, 178, 3), dtype=np.uint8) * 255
    total = np.concatenate([output, blank_hor, migsg], 0)
    total = cv2.resize(total, (178, 550))

    image_name = path + image_name
    # plt.imsave(image_name + "_g.png", np.uint8(g), format="png")
    # plt.imsave(image_name + "_ov_g.png", np.uint8(ov_g), format="png")
    #
    # plt.imsave(image_name + "_sg.png", np.uint8(sg), format="png")
    # plt.imsave(image_name + "_ov_sg.png", np.uint8(ov_sg), format="png")
    #
    # plt.imsave(image_name + "_ig.png", np.uint8(ig), format="png")
    # plt.imsave(image_name + "_ov_ig.png", np.uint8(ov_ig), format="png")
    #
    # plt.imsave(image_name + "_igsg.png", np.uint8(igsg), format="png")
    # plt.imsave(image_name + "_ov_igsg.png", np.uint8(ov_igsg), format="png")
    #
    plt.imsave(image_name + "_saliency.png", np.uint8(total), format="png")
    # plt.imsave(image_name + "_ov_migsg.png", np.uint8(ov_migsg), format="png")