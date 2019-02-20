from functions.adversarial import *
from functions.dip import *
from functions.generate_results import *

num_iter = 10001
image_dataset = ['labrador.jpg', 'snake.jpg', 'flamingo_animal.JPG', 'canoe.jpg',
                 'car_wheel.jpg', 'fountain.jpg', 'football_helmet.jpg', 'hourglass.jpg', 'refrigirator.jpg',
                 'knife.jpg', 'rope.jpeg']

# 'panda.jpg', 'peacock.jpg', 'F16_GT.png', 'monkey.jpg', 'zebra_GT.png', 'goldfish.jpg', 'whale.jpg',
#                  'dolphin.jpg', 'spider.jpg',

#for i in range(0,len(image_dataset)):
#    image_path = image_dataset[i]
#    image_name = '{}'.format(image_path.split('.')[0])
#    save_path_common = 'results/Adv_DIP/Skip_connections/{}'
#                
#    print("#############\n\nWorking on image: {}".format(image_name))           
#    adv, orig, pert = adversarial_examples("data/{}".format(image_path), method = "LLCI", eps = 100, show=False)
#    
#    for j in range(1,6):  
#        print("####\n\nTest {}".format(j))           
#
#        save_path=save_path_common.format('Adam/test{}'.format(j))
#       out = dip(adv,'skip{}'.format(j), 0.01, num_iter, save=True, plot=False, save_path = save_path)
#        generate_result_files(save_path, adv, orig, num_iter, image_name)

        #save_path=save_path_common.format('EntropySGD_std64/test{}'.format(j))
        #out = dip(adv,'skip{}'.format(j), 10, num_iter, 1/64., save=True, plot=False, save_path = save_path, OPTIMIZER = "EntropySGD")
        #generate_result_files(save_path, adv, orig, num_iter, image_name)       
        
for i in range(0, len(image_dataset)):
    image_path = image_dataset[i]
    image_name = '{}'.format(image_path.split('.')[0])
    save_path_common = 'results/Adv_DIP/Skip_connections/{}'

    print("#############\n\nWorking on image: {}".format(image_name))
    adv, orig, pert = adversarial_examples("data/{}".format(image_path), method="LLCI", eps=100, show=False)

    for j in range(6, 8):
        print("####\n\nTest {}".format(j))

        save_path = save_path_common.format('Adam/skip{}'.format(j))
        _ = dip(adv, 'skip{}'.format(j), 0.01, num_iter, save=True, plot=False, save_path=save_path)
        generate_result_files(save_path, adv, orig, num_iter, image_name)

        save_path = save_path_common.format('EntropySGD_std64/skip{}'.format(j))
        _ = dip(adv, 'skip{}'.format(j), 10, num_iter, 1/64., save=True, plot=False, save_path=save_path, OPTIMIZER="EntropySGD")
        generate_result_files(save_path, adv, orig, num_iter, image_name)
        
#for i in range(0,len(image_dataset)):
#    image_path = image_dataset[i]
#    image_name = '{}'.format(image_path.split('.')[0])
#    save_path_common = 'results/Adv_DIP/Std_investigation/{}'
#    print("#############\n\nWorking on image: {}".format(image_name))
#    adv, orig, pert = adversarial_examples("data/{}".format(image_path), method = "LLCI", eps = 100, show=False)
    
#    save_path=save_path_common.format('EntropySGD')
#    out = dip(adv,'complex', 10, num_iter, 1/64., save=True, plot=False, save_path = save_path, OPTIMIZER = "EntropySGD")
#    generate_result_files(save_path, adv, orig, num_iter, image_name)
    
    
    
    
    
