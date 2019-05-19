from functions.classification import *
import matplotlib.pyplot as plt
import cv2

f = open('Wnid_convertion.txt', 'r')
contents = f.read()
contents = contents.split()

wnid = [contents[i] for i in range(0,400,2)]
classid = [contents[i] for i in range(1,400,2)]

f = open("Class_labels2.txt", "w+")
counter = 0
for i in range(1, 200):
    for j in range(10):
        try:
            image = cv2.imread('ImageNet_dataset/' + wnid[i] + '/Image_{}.png'.format(j))[..., ::-1]
            image = cv2.resize(image, (299, 299))
            _, ranks = classification(image, sort=True, show=False, model_name='InceptionV3', cuda=False)
            if ranks[0, 0] == int(classid[i]):
                plt.imsave('correctly_classified_dataset2/Image_{}.png'.format(counter), image, format ='png')
                counter = counter + 1
                f = open("Class_labels2.txt", "a")
                f.write("{}\n".format(classid[i]))
            print(counter)
        except:
            print('Bad Image')

print('Images correctly classified = {}'.format(counter))