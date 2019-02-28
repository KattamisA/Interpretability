from functions.adversarial import *
import matplotlib.pyplot as plt


image_path = 'panda.jpg'

# adv, _, pert = adversarial_examples("data/{}".format(image_path), method="JSMA", eps=1, show=True, cuda=True)

adv, orig, pert = adversarial_examples("data/{}".format(image_path), method="JSMA", eps=100, show=True, num_iter=30, cuda=True)

plt.imsave("data/{}_JSMA.png".format(image_path.split('.')[0]), adv, format='png')


adv = adv.astype(np.float32)
orig = orig.astype(np.float32)
diff = np.absolute(adv-orig)
plt.imsave("data/{}_JSMA_pert.png".format(image_path), diff.astype(np.int8)*20, format='png')
print(np.count_nonzero(diff))
print(diff.astype(np.int8)*20)


