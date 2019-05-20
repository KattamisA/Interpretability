from skimage.measure import compare_psnr
import cv2
import numpy as np


### Observing multiple images
image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']
PSNR = np.ones([20, 1])
for i in range(len(image_dataset)):
    image_path = image_dataset[i]
    image_name = '{}'.format(image_path.split('.')[0])
    orig = cv2.imread('data/{}'.format(image_path))[..., ::-1]
    orig = cv2.resize(orig, (224,224))
    img_noisy = cv2.imread('results/Denoising/dataset/{}_noisy.png'.format(image_name))[..., ::-1]
    PSNR[i, 0] = compare_psnr(orig/255.0, img_noisy/255.0).astype(np.float32)

print(np.average(PSNR))