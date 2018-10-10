import os
import numpy as np 
from skimage import data
from skimage import color
import pickle

images = []
labels = []

directory = './training'

for d in os.listdir(directory):
    labels.append(d)
    tmp_imgs = []
    for img_path in os.listdir(os.path.join(directory, d)):
        img = data.imread("{}/{}/{}".format(directory, d, img_path))
        # img = color.rgb2gray(img)
        tmp_imgs.append(img)
    images.append(tmp_imgs)

labels = np.array(labels, dtype=np.uint8)
images = np.array(images, dtype=np.float32)

# with open('data.pickle', 'wb') as f:
#     pickle.dump((images, labels), f)
np.savez('data.npz', images=images, labels=labels)