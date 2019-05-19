from __future__ import print_function

from functions.adversarial import *
from functions.dip import *

f = open("Class_labels2.txt", 'r')
classids = f.read()
classids= classids.split()

w = open("Results.txt",'w+')

no_defence_rec = 0
defence_rec = 0
num = 0
for i in range(0, len(classids), 3):
    num = num + 1
    adv, _, _ = adversarial_examples("correctly_classified_dataset2/Image_{}.png".format(i), eps=2/0.226, show=False, model_name = 'inception_v3')
    _, ranks_adv = classification(adv, sort=True, show=False, model_name='inception_v3', cuda=True)
    # plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')
    output = dip(adv, 'complex', 0.01, 501, save=False, plot=False)
    _, ranks_rec = classification(output*255.0, sort=True, show=False, model_name='inception_v3', cuda=True)
    if ranks_adv[0,0] == int(classids[i]):
        no_defence_rec = no_defence_rec + 1
    if ranks_rec[0,0] == int(classids[i]):
        defence_rec = defence_rec + 1
    print("Results after {}: Defence: {} --- No defence: {}".format(num, float(defence_rec)/num, float(no_defence_rec)/num), end='\r')

w.write("FGSM2: {} {}\n".format(2.0*float(defence_rec)/len(classids), 2.0*float(no_defence_rec)/len(classids)))






# for i in range(len(classids)):
#
#     adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=5/0.226, show=False)
#     plt.imsave("data/adversarial_defence_datasets/FGSM5/adv_Image_{}.png".format(i), adv, format='png')
# for i in range(len(classids)):
#
#     adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=10/0.226, show=False)
#     plt.imsave("data/adversarial_defence_datasets/FGSM10/adv_Image_{}.png".format(i), adv, format='png')


