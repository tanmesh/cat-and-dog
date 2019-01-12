import os
import re

import cv2


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]



def prepare_data(list_of_images_path, img_width, img_height):
    x = []  # images as arrays
    y = []  # labels
    for image_path in list_of_images_path:
        read_image = cv2.imread(image_path)
        tmp = cv2.resize(read_image, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        x.append(tmp)

    for i in list_of_images_path:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)

    return x, y


def split_data():
    train_dir = '/Users/tanmesh/dev/cat_and_dog/dataset/train/'
    test_dir = '/Users/tanmesh/dev/cat_and_dog/dataset/test/'
    train_images_dogs_cats = [train_dir + i for i in os.listdir(train_dir)]  # use this for full dataset
    test_images_dogs_cats = [test_dir + i for i in os.listdir(test_dir)]
    train_images_dogs_cats.sort(key=natural_keys)
    train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13800]
    test_images_dogs_cats.sort(key=natural_keys)

    return train_images_dogs_cats, test_images_dogs_cats

