import cv2
import numpy as numpy
import os
import glob

training_path = './training_data'
testing_path = './testing_data'

for folder in os.listdir(training_path):
    os.makedirs("{}/{}".format('./training', folder))
    for img_path in glob.glob('{}/{}/*.png'.format(training_path, folder))[:20]:
        image = cv2.imread(img_path)
        resized_img = cv2.resize(image, (64, 64))
        cv2.imwrite("{}/{}/{}".format('./training', folder, os.path.basename(img_path)), resized_img)
        print(img_path)

for folder in os.listdir(testing_path):
    os.makedirs("{}/{}".format('./testing', folder))
    for img_path in glob.glob('{}/{}/*.png'.format(testing_path, folder))[:2]:
        image = cv2.imread(img_path)
        resized_img = cv2.resize(image, (64, 64))
        cv2.imwrite("{}/{}/{}".format('./testing', folder, os.path.basename(img_path)), resized_img)
        print(img_path)