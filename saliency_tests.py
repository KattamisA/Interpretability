from functions.generate_saliency_maps import generate_saliency_maps
from functions.adversarial import *
from functions.classification import *
import matplotlib.pyplot as plt

image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'knife.jpg', 'rope.jpeg']
# f = open('results/Saliency/adversarial/llci/LLCI_classification.txt', "w+")
#q = open('results/Saliency/adversarial/llci/LLCI_classification_inception_v3.txt', "w+")
for i in range(len(image_dataset)):
    image = image_dataset[i]
    print('###### Working on image: ' + image.split('.')[0])
    #adv, orig, pert = adversarial_examples("data/{}".format(image), method = "LLCI", eps = 100, alpha=50, show=False)
    img = cv2.imread('data/' + image)[..., ::-1]
    _, ranks = classification(img, sort=True, cuda=True)
    target_class = ranks[0,0]
    name = image.split('.')[0]
    print('## Dual polarity test ##')
    image = name + '_llci_adversarial.png'
    save_path = 'results/Saliency/adversarial/llci/Dual_polarity_tests'
    img_path = 'results/Saliency/adversarial/llci/' + image
    generate_saliency_maps(save_path, img_path, name, model_type='resnet18', cuda=True, target_label=target_class,
                        top_percentile=95, bottom_percentile=50, mask_mode=True, stdev_spread=0.01, dual=True)

    print('## Smoothgrad with std = 0.05 ##')
    save_path = 'results/Saliency/adversarial/llci/Smoothgrad_std005'
    generate_saliency_maps(save_path, img_path, name, model_type='resnet18', cuda=True, target_label=target_class,
                        top_percentile=95, bottom_percentile=50, mask_mode=True, stdev_spread=0.05, dual=False)

    print('## Smoothgrad with std = 0.01 ##')
    save_path = 'results/Saliency/adversarial/llci/Smoothgrad_std001'
    generate_saliency_maps(save_path, img_path, name, model_type='resnet18', cuda=True, target_label=target_class,
                        top_percentile=95, bottom_percentile=50, mask_mode=True, stdev_spread=0.01, dual=False)

    print('## Inception v3 ##')
    image = name + '_inception_v3.png'
    save_path = 'results/Saliency/adversarial/fgsm/Inception_v3'
    img_path = 'results/Saliency/adversarial/fgsm/Inception_v3/' + image
    generate_saliency_maps(save_path, img_path, name, model_type='resnet18', cuda=True, target_label=target_class,
                        top_percentile=99, bottom_percentile=10, mask_mode=True, stdev_spread=0.15, dual=False)

    print('## Resnet 18 ##')
    image = name + '.png'
    save_path = 'results/Saliency/adversarial/fgsm/Resnet18'
    img_path = 'results/Saliency/adversarial/fgsm/Resnet18/' + image
    generate_saliency_maps(save_path, img_path, name, model_type='resnet18', cuda=True, target_label=target_class,
                        top_percentile=99, bottom_percentile=10, mask_mode=True, stdev_spread=0.15, dual=False)
    print('\n')

    # img = cv2.imread(save_path + '/' + image)[..., ::-1]
    # confidence, ranks = classification(img, sort=True, show=False, model_name='resnet18', cuda=True)
    # Class = classes[int(ranks[0, 0])].split(',')[0]
    # f.write("{:<30}{:>10.6f}\n".format(Class, confidence[0, 0]))


# image_dataset2 = ['it_{}.png'.format(100*i) for i in range(0, 11)]
# image_dataset2.extend(['it_{}.png'.format(200*i) for i in range(6, 51)])
#
# for i in range(len(image_dataset2)):
#      image = image_dataset2[i]
#
#      print('###### Working on image: ' + image.split('.')[0])
#      generate_saliency_maps('results/Adv_DIP/Multiple_images/knife', image, model_type='resnet18', cuda=True,
#                             top_percentile=99, bottom_percentile=10, mask_mode=True)
#
#      #generate_saliency_maps('results/Adv_DIP/Multiple_images/knife', "adv_knife.png", model_type='resnet18', cuda=True,
#      #                      top_percentile=99, bottom_percentile=0, mask_mode=True, target_label_index=class_index)
#      print('\n')

# image_dataset = ['zebra_GT.png', 'knife.jpg']
#
# for image in image_dataset:
#     adv, orig, pert = adversarial_examples("data/{}".format(image), method="LLCI", eps=100, show=False,
#                                            model_name='resnet18', cuda=False)
#     name = image.split('.')[0]
#     plt.imsave('results/Saliency/adversarial/' + name + '_adversarial.png', adv, format='png')
#     generate_saliency_maps('results/Saliency/adversarial', name+'_adversarial.png', model_type='resnet18', cuda=True,
#                            top_percentile=99, bottom_percentile=10, mask_mode=True)
#     print('\n')
