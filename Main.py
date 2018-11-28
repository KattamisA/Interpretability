from functions.adversarial import *
from functions.dip import dip
from functions.classification import *
import matplotlib.pyplot as plt
import cv2
import argparse
import numpy as np
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/goldfish.jpg', help='path to image')
parser.add_argument('--model', type=str, default='resnet18',
                    choices=['resnet18', 'inception_v3', 'resnet50'],
                    required=False, help="Which network?")
parser.add_argument('--y', type=int, required=False, help='Label')

device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
print(device)

args = parser.parse_args()
image_path = args.img
model_name = args.model
y_true = args.y
def nothing(x):
	pass
#window_adv = 'adversarial image'
#cv2.namedWindow(window_adv)

orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)
#img = img[..., ::-1] # RGB to BGR
img /= 255.0

while True:
	out = dip(img,num_iter=2000,save=True,PLOT=False,save_path='results/Tests')
	key = cv2.waitKey(500) & 0xFF
	if key == 27:
		break
	elif key == ord('s'):
		cv2.imwrite('img_adv.png', adv)
		cv2.imwrite('perturbation.png', perturbation)
			
print()
cv2.destroyAllWindows()

