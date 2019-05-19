from functions.adversarial import *
import cv2

image = cv2.imread("ImageNet_dataset/n02226429/Image_4.png")[..., ::-1]
_, ranks_adv = classification(image, sort=True, show=False, model_name='resnet18', cuda=False)
print(ranks_adv[0, 0])
adv, _, _ = adversarial_examples("ImageNet_dataset/n02226429/Image_4.png", eps=0.01, show=False)
_, ranks_adv = classification(adv, sort=True, show=False, model_name='resnet18', cuda=False)
print(ranks_adv[0, 0])

# classification(image)
# adv, orig, pert = adversarial_examples("data/goldfish.jpg", eps=2/0.226)
# adv = adv.astype(np.int8)
# print(pert)
# print(np.max(adv))
# print(np.min(adv))