from functions.adversarial import *
import cv2

f = open("Class_labels.txt",'r')
contents = f.read()
contents= contents.split()
print(len(contents))
for i in range(5):
    print(i)
    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=2/0.226, show=False)
    plt.imsave("data/adversarial_defence_datasets/FGSM2/adv_Image_{}.png".format(i), adv, format='png')

    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=5/0.226, show=False)
    plt.imsave("data/adversarial_defence_datasets/FGSM5/adv_Image_{}.png".format(i), adv, format='png')

    adv, _, _ = adversarial_examples("correctly_classified_dataset/Image_{}.png".format(i), eps=10/0.226, show=False)
    plt.imsave("data/adversarial_defence_datasets/FGSM10/adv_Image_{}.png".format(i), adv, format='png')


