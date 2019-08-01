from __future__ import print_function

from functions.adversarial import *
from functions.dip import *
import cv2

f = open("data/ImageNet_Dataset/Class_labels.txt", 'r')
classids = f.read()
classids= classids.split()

defence_rec = 0
num = 0
print('\nJSMA - 10')
for i in range(1242, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False, method='JSMA', cuda=True)
    output = dip(adv, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/JSMA10', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f}".format(num, float(defence_rec)/num), end='\r')

# w = open("Results_JSMA10_dip1000.txt",'w+')
# w.write("JSMA10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_de''fence_rec)/num))