from __future__ import print_function

from functions.adversarial import *
from functions.dip import *

f = open("Class_labels.txt", 'r')
classids = f.read()
classids= classids.split()


no_defence_rec = 0
defence_rec = 0
num = 0
print('\nFGSM - 2')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=2, show=False)
    _, ranks_adv = classification(adv, sort=True, show=False, model_name='resnet18', cuda=True)
    # plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')
    output = dip(adv, 'complex', 0.01, 501, save=False, plot=False)
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_adv[0,0] == int(classids[i]):
        no_defence_rec = no_defence_rec + 1
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f} --- No defence: {:>7.3f}".format(num, float(defence_rec)/num, float(no_defence_rec)/num), end='\r')

w = open("Results_FGSM2.txt",'w+')
w.write("FGSM2: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))



no_defence_rec = 0
defence_rec = 0
num = 0
print('\nFGSM - 5')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=5, show=False)
    _, ranks_adv = classification(adv, sort=True, show=False, model_name='resnet18', cuda=True)
    # plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')
    output = dip(adv, 'complex', 0.01, 501, save=False, plot=False)
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_adv[0,0] == int(classids[i]):
        no_defence_rec = no_defence_rec + 1
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f} --- No defence: {:>7.3f}".format(num, float(defence_rec)/num, float(no_defence_rec)/num), end='\r')

w = open("Results_FGSM5.txt",'w+')
w.write("FGSM5: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))



no_defence_rec = 0
defence_rec = 0
num = 0
print('\nFGSM - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False)
    _, ranks_adv = classification(adv, sort=True, show=False, model_name='resnet18', cuda=True)
    # plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')
    output = dip(adv, 'complex', 0.01, 501, save=False, plot=False)
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_adv[0,0] == int(classids[i]):
        no_defence_rec = no_defence_rec + 1
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f} --- No defence: {:>7.3f}".format(num, float(defence_rec)/num, float(no_defence_rec)/num), end='\r')

w = open("Results_FGSM10.txt",'w+')
w.write("FGSM10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))




no_defence_rec = 0
defence_rec = 0
num = 0
print('\nBI - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False, method='BI')
    _, ranks_adv = classification(adv, sort=True, show=False, model_name='resnet18', cuda=True)
    # plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')
    output = dip(adv, 'complex', 0.01, 501, save=False, plot=False)
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_adv[0,0] == int(classids[i]):
        no_defence_rec = no_defence_rec + 1
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f} --- No defence: {:>7.3f}".format(num, float(defence_rec)/num, float(no_defence_rec)/num), end='\r')

w = open("Results_BI10.txt",'w+')
w.write("BI10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))




no_defence_rec = 0
defence_rec = 0
num = 0
print('\nLLCI - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False, method='LLCI')
    _, ranks_adv = classification(adv, sort=True, show=False, model_name='resnet18', cuda=True)
    # plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')
    output = dip(adv, 'complex', 0.01, 501, save=False, plot=False)
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_adv[0,0] == int(classids[i]):
        no_defence_rec = no_defence_rec + 1
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f} --- No defence: {:>7.3f}".format(num, float(defence_rec)/num, float(no_defence_rec)/num), end='\r')

w = open("Results_LLCI10.txt",'w+')
w.write("LLCI10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))



no_defence_rec = 0
defence_rec = 0
num = 0
print('\nJSMA - 10')
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=10, show=False, method='JSMA', cuda=True)
    _, ranks_adv = classification(adv, sort=True, show=False, model_name='resnet18', cuda=True)
    # plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')
    output = dip(adv, 'complex', 0.01, 501, save=False, plot=False)
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='resnet18', cuda=True)
    if ranks_adv[0,0] == int(classids[i]):
        no_defence_rec = no_defence_rec + 1
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {:>3}: Defence: {:>7.3f} --- No defence: {:>7.3f}".format(num, float(defence_rec)/num, float(no_defence_rec)/num), end='\r')

w = open("Results_JSMA10.txt",'w+')
w.write("JSMA10: {:>10.5f} {:>10.5f}\n".format(float(defence_rec)/num, float(no_defence_rec)/num))