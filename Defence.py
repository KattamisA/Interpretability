from functions.adversarial import *
from functions.dip import *
from functions.classification import *
import matplotlib.pyplot as plt
import cv2

f = open('Class_labels.txt','r')
classid = f.read()
classid = classid.split()

for












# z = []
# q=0.0
# p=0.0
# for i in range(len(image_dataset)):
#     image_path = image_dataset[i]
#     image_name = '{}'.format(image_path.split('.')[0])
#     print(image_name)
#     orig = cv2.imread("data/" + image_path)[..., ::-1]
#     _, ranks = classification(orig, sort=True, show=False, model_name='resnet18', cuda=True)
#     orig_rank = ranks[0, 0]
#     adv = cv2.imread("results/adversarial_examples/Examples/FGSM_eps1/" + image_name + "_FGSM_eps1.png")[..., ::-1]
#     _, ranks = classification(adv , sort=True, show=False, model_name='resnet18', cuda=True)
#     if ranks[0, 0] == orig_rank:
#         p = p + 1
#     output = dip(adv, 'complex', 0.01, 301, save=False, plot=False, name=image_name)
#     _, ranks = classification(output * 255, sort=True, show=False, model_name='resnet18', cuda=True)
#     if ranks[0,0] == orig_rank:
#         q = q + 1
# print("\n\nResults: Adversary - {}      Recovered - {}".format(p/20.0,q/20.0))
# z.append([q / 20.0, p / 20.0])
# print(z)