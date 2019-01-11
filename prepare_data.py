import os
import re

import cv2


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def update(file_name):
    str_train = '/Users/tanmesh/dev/cat_and_dog/dataset/train'
    str_test = '/Users/tanmesh/dev/cat_and_dog/dataset/test'
    new_file = []
    for name in file_name:
        if name.find(str_train) != -1:
            new_file.append(name.replace(str_train, ''))
        if name.find(str_test) != -1:
            new_file.append(name.replace(str_test, ''))
    return new_file


def data(list_of_images, img_width, img_height):
    x = []  # images as arrays
    y = []  # labels
    cnt_exp = 0
    for image in list_of_images:
        try:
            tmp = cv2.resize(cv2.imread(image), (img_width, img_height), interpolation=cv2.INTER_CUBIC)
            x.append(tmp)
        except Exception:
            # print("There is some exception")
            cnt_exp += 1
            pass

    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        # else:
        # print('neither cat nor dog name present in images')

    print("cnt_exp = " + str(cnt_exp))
    return x, y


def prepare_data():
    img_width = 150
    img_height = 150
    train_dir = '/Users/tanmesh/dev/cat_and_dog/dataset/train'
    test_dir = '/Users/tanmesh/dev/cat_and_dog/dataset/test'
    train_images_dogs_cats = [train_dir + i for i in os.listdir(train_dir)]  # use this for full dataset
    test_images_dogs_cats = [test_dir + i for i in os.listdir(test_dir)]

    train_images_dogs_cats = update(train_images_dogs_cats)
    test_images_dogs_cats = update(test_images_dogs_cats)

    train_images_dogs_cats.sort(key=natural_keys)
    train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13800]
    test_images_dogs_cats.sort(key=natural_keys)

    x, y = data(train_images_dogs_cats, img_width, img_height)

    print("x = " + str(len(x)))
    print("y = " + str(len(y)))
    return x, y


prepare_data()
