from functions.adversarial import *
import matplotlib.pyplot as plt


image_path = 'panda.jpg'

adv, _, pert = adversarial_examples("data/{}".format(image_path), method="FGSM", eps=100, show=True)
print(pert)

# plt.imsave("data/{}_JSMA.png".format(image_path), adv, format='png')

# adv = cv2.imread("data/{}_JSMA.png".format(image_path))[..., ::-1]
# orig = cv2.imread("data/{}".format(image_path))[..., ::-1]
# orig = cv2.resize(orig, (224, 224))
# plt.imsave("data/{}_JSMA_pert.png".format(image_path), (adv-orig)*50.0, format='png')
# print(adv-orig)


