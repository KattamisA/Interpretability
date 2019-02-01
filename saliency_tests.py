from functions.generate_saliency_maps import generate_saliency_maps

#image_dataset = ['labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
#                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
#                 'knife.jpg', 'rope.jpeg']

#'panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg','dolphin.jpg', 'spider.jpg',
image_dataset = ['it_{}.png'.format(100*i) for i in range(0,101)]

for i in range(len(image_dataset)):
    image = image_dataset[i]
    print('###### Working on image: ' + image.split('.')[0])
    generate_saliency_maps('results/Adv_DIP/Multiple_images/zebra_GT', image, model_type='resnet18', cuda=True, top_percentile=99, bottom_percentile=0, mask_mode=True)
    print('\n')

