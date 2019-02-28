from functions.adversarial import *
import matplotlib.pyplot as plt


image_path = 'panda.jpg'

# adv, _, pert = adversarial_examples("data/{}".format(image_path), method="JSMA", eps=1, show=True, cuda=True)

adv, orig, pert = adversarial_examples("data/{}".format(image_path), method="JSMA", eps=100, show=True, num_iter=125, cuda=True)

plt.imsave("data/{}_JSMA.png".format(image_path.split('.')[0]), adv, format='png')

plt.imsave("data/{}_JSMA_pert.png".format(image_path), (adv-orig)*50.0, format='png')
print(adv-orig)


