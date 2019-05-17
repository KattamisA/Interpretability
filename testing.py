import numpy as np
import cv2
import matplotlib.pyplot as plt
#
# image_dataset = ['it_{}.png'.format(200*i) for i in range(0, 99)]
#
# input = cv2.imread("data/panda.jpg")[..., ::-1]
# input = cv2.resize(input, (256, 256))
# plt.imsave("data/panda.png", np.uint8(input), format="png")

# for i in image_dataset:
#     output = cv2.imread("results/DIP_aero/DIP_output/{}".format(i))[..., ::-1]
#     result = output - input
#     #result = np.absolute(result)
#     plt.imsave("results/DIP_aero/" + i, np.uint8(result), format="png")
for i in range(2,3):
    print(i)