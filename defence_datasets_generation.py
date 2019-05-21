from __future__ import print_function

from functions.adversarial import *
from functions.dip import *
import cv2

f = open("data/ImageNet_Dataset/Class_labels.txt", 'r')
classids = f.read()
classids= classids.split()


correct = 0
num = 0
print('\nClean')
for i in range(0, len(classids), 3):
    num = num + 1
    orig = cv2.imread("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i))[..., ::-1]
    orig = cv2.resize(orig, (224,224))
    output = dip(orig, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/Clean', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        correct = correct + 1
    print("Results after {:>3}: DIP: {:>7.3f}".format(num, float(correct)/num), end='\r')

# w = open("results/Defence/Results_clean.txt",'w+')
# w.write("Clean: {:>10.5f} {:>10.5f}\n".format(float(correct)/num, float(correct)/num))


defence_rec = 0
num = 0
print('\nFGSM - 2')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i), eps=2, show=False)
    output = dip(orig, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/FGSM2', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f}".format(num, float(defence_rec)/num), end='\r')

# w = open("results/Defence/Results_FGSM2_dip1000.txt",'w+')
# w.write("FGSM2: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))



defence_rec = 0
num = 0
print('\nFGSM - 5')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i), eps=5, show=False)
    output = dip(orig, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/FGSM5', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f}".format(num, float(defence_rec)/num), end='\r')

# w = open("results/Defence/Results_FGSM5_dip1000.txt",'w+')
# w.write("FGSM5: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))



defence_rec = 0
num = 0
print('\nFGSM - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False)
    output = dip(orig, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/FGSM10', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f}".format(num, float(defence_rec)/num), end='\r')

# w = open("results/Defence/Results_FGSM10_dip1000.txt",'w+')
# w.write("FGSM10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))




defence_rec = 0
num = 0
print('\nBI - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False, method='BI')
    output = dip(orig, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/BI10', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f}".format(num, float(defence_rec)/num), end='\r')

# w = open("results/Defence/Results_BI10_dip1000.txt",'w+')
# w.write("BI10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))



defence_rec = 0
num = 0
print('\nLLCI - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False, method='LLCI')
    output = dip(orig, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/LLCI10', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f}".format(num, float(defence_rec)/num), end='\r')

# w = open("Results_LLCI10_dip1000.txt",'w+')
# w.write("LLCI10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))


defence_rec = 0
num = 0
print('\nJSMA - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("data/ImageNet_Dataset/correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False, method='JSMA', cuda=True)
    output = dip(orig, 'complex', 0.01, 1001, plot=False, save=True, save_path = 'results/Defence/JSMA10', name = "Image_{}".format(i))
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f}".format(num, float(defence_rec)/num), end='\r')

# w = open("Results_JSMA10_dip1000.txt",'w+')
# w.write("JSMA10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))