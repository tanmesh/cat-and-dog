import os
import re
from cv2 import INTER_CUBIC, imread, resize


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def data(list_of_images, img_width, img_height):
    x = []  # images as arrays
    y = []  # labels

    for image in list_of_images:
        try:
            tmp = resize(imread(image), (img_width, img_height), interpolation=INTER_CUBIC)
            x.append(tmp)
        except Exception:
            pass


    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        # else:
        # print('neither cat nor dog name present in images')

    return x, y


def prepare_data():
    img_width = 150
    img_height = 150
    train_dir = '/Users/tanmesh/dev/cat_and_dog/dataset/train'
    test_dir = '/Users/tanmesh/dev/cat_and_dog/dataset/test'
    train_images_dogs_cats = [train_dir + i for i in os.listdir(train_dir)]  # use this for full dataset
    test_images_dogs_cats = [test_dir + i for i in os.listdir(test_dir)]

    train_images_dogs_cats.sort(key=natural_keys)
    train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13800]

    test_images_dogs_cats.sort(key=natural_keys)

    x, y = data(train_images_dogs_cats, img_width, img_height)

    return x, y

