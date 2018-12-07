import cv2
import numpy as np
import torch
from functions.classification import *

def generate_result_files(path, adv, orig, num_iter):
    ## Find original class
    P, R = classification(orig, model_name = 'resnet18', sort = True, show=False)
    original_class = R[0,0]
    ## Find final set of classes
    P, R = classification(adv, model_name = 'resnet18', sort = True, show=False)
    final_classes = R[0,0:5]
    
    num_images = (num_iter-1)/100 + 1

    Confidence = np.ones([num_images,6])
    Ranks_matrix = np.ones([num_images,5])

    for i in range(num_images):
        image = cv2.imread('{}/it_{}.png'.format(path,i*100))[..., ::-1]
        image = cv2.resize(orig, (256, 256))
        img = orig.copy().astype(np.float32)
        Probs, Ranks = classification(img, model_name = 'resnet18', sort = False, show = False)
        Probs_np = torch_to_np(Probs)
        Confidence[i,0] = Probs_np[original_class]
        P , Ranking = Probs.sort(descending=True)
        Ranking_np = torch_to_np(Ranking)
        for j in range(5):
            Confidence[i,j+1] = Probs_np[final_classes[j]]
            Ranks_matrix[i,j] = Ranking_np[j]

    np.savetxt('{}/Confidences.txt'.format(path), Confidence)
    np.savetxt('{}/Ranks.txt'.format(path), Ranks_matrix)
    
    print('\n Results have been generated and stored in {}'.format(path))