from functions.saliency import *

images = 'panda.jpg'
# generate_saliency_maps('results/Adv_DIP/Multiple_images/panda', img, 10001)
generate_saliency_maps('data', images, cuda=True, top_percentile=99, bottom_percentile=50)

