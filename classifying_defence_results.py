from __future__ import print_function

from functions.adversarial import *
from functions.dip import *
import numpy as np
import cv2

f = open("data/ImageNet_Dataset/Class_labels.txt", 'r')
classids = f.read()
classids= classids.split()

defence = np.zeros((4, 1))
num = 0
print('\nClean')
for i in range(0, len(classids), 3):
    num = num + 1
    for j in range(250,1001,250):
        image = cv2.imread("results/Defence/FGSM2/Image_{}_{}it.png".format(i, j))[..., ::-1]
        _, ranks_rec = classification(image, sort=True, show=False, model_name='resnet18', cuda=True)
        if ranks_rec[0, 0] == int(classids[i]):
            defence[int(j/250)-1] = defence[int(j/250)-1] + 1
    print("Clean {:>3} --- 250it: {:>7.3f}    500it: {:>7.3f}    750it: {:>7.3f}    100it: {:>7.3f}"
          .format(num, float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num), end='\r')

w = open("results/Defence/Results_FGSM2.txt",'w+')
w.write("Clean: {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num))

defence = np.zeros((4, 1))
num = 0
print('\nClean')
for i in range(0, len(classids), 3):
    num = num + 1
    for j in range(250,1001,250):
        image = cv2.imread("results/Defence/FGSM5/Image_{}_{}it.png".format(i, j))[..., ::-1]
        _, ranks_rec = classification(image, sort=True, show=False, model_name='resnet18', cuda=True)
        if ranks_rec[0, 0] == int(classids[i]):
            defence[int(j/250)-1] = defence[int(j/250)-1] + 1
    print("Clean {:>3} --- 250it: {:>7.3f}    500it: {:>7.3f}    750it: {:>7.3f}    100it: {:>7.3f}"
          .format(num, float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num), end='\r')

w = open("results/Defence/Results_FGSM5.txt",'w+')
w.write("Clean: {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num))

defence = np.zeros((4, 1))
num = 0
print('\nClean')
for i in range(0, len(classids), 3):
    num = num + 1
    for j in range(250,1001,250):
        image = cv2.imread("results/Defence/FGSM10/Image_{}_{}it.png".format(i, j))[..., ::-1]
        _, ranks_rec = classification(image, sort=True, show=False, model_name='resnet18', cuda=True)
        if ranks_rec[0, 0] == int(classids[i]):
            defence[int(j/250)-1] = defence[int(j/250)-1] + 1
    print("Clean {:>3} --- 250it: {:>7.3f}    500it: {:>7.3f}    750it: {:>7.3f}    100it: {:>7.3f}"
          .format(num, float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num), end='\r')

w = open("results/Defence/Results_FGSM10.txt",'w+')
w.write("Clean: {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num))

defence = np.zeros((4, 1))
num = 0
print('\nClean')
for i in range(0, len(classids), 3):
    num = num + 1
    for j in range(250,1001,250):
        image = cv2.imread("results/Defence/BI10/Image_{}_{}it.png".format(i, j))[..., ::-1]
        _, ranks_rec = classification(image, sort=True, show=False, model_name='resnet18', cuda=True)
        if ranks_rec[0, 0] == int(classids[i]):
            defence[int(j/250)-1] = defence[int(j/250)-1] + 1
    print("Clean {:>3} --- 250it: {:>7.3f}    500it: {:>7.3f}    750it: {:>7.3f}    100it: {:>7.3f}"
          .format(num, float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num), end='\r')

w = open("results/Defence/Results_BI10.txt",'w+')
w.write("Clean: {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num))

defence = np.zeros((4, 1))
num = 0
print('\nClean')
for i in range(0, len(classids), 3):
    num = num + 1
    for j in range(250,1001,250):
        image = cv2.imread("results/Defence/LLCI10/Image_{}_{}it.png".format(i, j))[..., ::-1]
        _, ranks_rec = classification(image, sort=True, show=False, model_name='resnet18', cuda=True)
        if ranks_rec[0, 0] == int(classids[i]):
            defence[int(j/250)-1] = defence[int(j/250)-1] + 1
    print("Clean {:>3} --- 250it: {:>7.3f}    500it: {:>7.3f}    750it: {:>7.3f}    100it: {:>7.3f}"
          .format(num, float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num), end='\r')

w = open("results/Defence/Results_LLCI10.txt",'w+')
w.write("Clean: {:>10.5f} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(float(defence[0])/num, float(defence[1])/num, float(defence[2])/num, float(defence[3])/num))