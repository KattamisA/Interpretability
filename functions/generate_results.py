import cv2
import numpy as np
from functions.classification import classification
from functions.utils.common_utils import *
from skimage.measure import compare_psnr


def generate_result_files(path, adv, orig, num_iter, name, cuda=False, model = 'resnet18'):
    ## Find original class
    P, R = classification(orig, model_name = 'resnet18', sort = True, show=False)
    original_class = R[0, 0]
    original_confidence = P[0, 0].detach().numpy()
    ## Find final set of classes
    _, R = classification(adv, model_name = 'resnet18', sort = True, show=False)
    final_classes = R[0, 0:5]
    
    num_images = int((num_iter-1)/100 + 1)

    Confidence = np.ones([num_images, 6])
    Ranks_matrix = np.ones([num_images, 5])
    PSNR = np.ones([num_images, 1])

    for i in range(num_images):
        loaded_image = cv2.imread('{}/it_{}.png'.format(path,i *100))[..., ::-1]
        loaded_image = cv2.resize(loaded_image, (224, 224))
        img = loaded_image.copy().astype(np.float32)
        Probs, _ = classification(img, model_name=model, sort=False, show=False, cuda=cuda)
        Probs_np = torch_to_np(Probs)
        Confidence[i, 0] = Probs_np[original_class]
        _, Ranking = Probs.sort(descending=True)
        Ranking_np = torch_to_np(Ranking)
        print(np.shape(img))
        PSNR[i, 1] = compare_psnr(orig, img)
        for j in range(5):
            Confidence[i, j+1] = Probs_np[final_classes[j]]
            Ranks_matrix[i, j] = Ranking_np[j]
            
    normalised_confidence = Confidence[:, 0]/original_confidence
    np.savetxt('{}/{}_Confidences.txt'.format(path, name), Confidence)
    np.savetxt('{}/{}_Ranks.txt'.format(path, name), Ranks_matrix)
    np.savetxt('{}/{}_Normalised.txt'.format(path, name), normalised_confidence)
    np.savetxt('{}/{}_PSNR.txt'.format(path, name), PSNR)

    print('\nResults have been generated and stored in {}'.format(path))