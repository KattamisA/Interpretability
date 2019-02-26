from functions.dip import *
from functions.generate_results import *

### Observing multiple images
num_iter = 5001
image_dataset = ['panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
                 'dolphin.jpg', 'spider.jpg', 'labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'rope.jpeg', 'knife.jpg']

for i in range(len(image_dataset)):
    image_path = image_dataset[i]
    image_name = '{}'.format(image_path.split('.')[0])

    orig = cv2.imread("data/{}".format(image_path))[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)
    std = 15
    img_noisy = img + std*np.random.randn(224,224,3)
    img_noisy = np.clip(img_noisy,0,255).astype(np.uint8)
    
    save_path = 'results/Denoising/Baseline'
    _ = dip(img_noisy, 'complex', 0.01, num_iter, save=True, plot=False, save_path=save_path, name=image_name)
    generate_result_files(save_path, img_noisy, img, num_iter, image_name)

    # save_path='results/Denoising/Multiple_images/EntropySGD/{}'.format(image_name)[0])
    # out = dip(img_noisy, num_iter=num_iter, save=True, plot=False, save_path = save_path, arch='complex', OPTIMIZER = "EntropySGD", LR = 1)
    # generate_result_files(save_path, img_noisy, img, num_iter)
    
    

