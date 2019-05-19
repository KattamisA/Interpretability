from functions.adversarial import *
import cv2

image = cv2.imread("data/goldfish.jpg")[..., ::-1]

adv, orig, pert = adversarial_examples("data/goldfish.jpg", eps=1/0.226)
adv = adv.astype(np.int8)
print(pert)
print(np.max(adv))
print(np.min(adv))