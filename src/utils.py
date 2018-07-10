# ---------------------------------------------------------
# Tensorflow CycleGAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import scipy.misc
import numpy as np


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()


def transform(imgs):
    return imgs / 127.5 - 1.0


def inverse_transform(imgs):
    return (imgs + 1.) / 2.


def preprocess_pair(img_a, img_b, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_a = scipy.misc.imresize(img_a, [fine_size, fine_size])
        img_b = scipy.misc.imresize(img_b, [fine_size, fine_size])
    else:
        img_a = scipy.misc.imresize(img_a, [load_size, load_size])
        img_b = scipy.misc.imresize(img_b, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_a = img_a[h1:h1 + fine_size, w1:w1 + fine_size]
        img_b = img_b[h1:h1 + fine_size, w1:w1 + fine_size]

        if flip and np.random.random() > 0.5:
            img_a = np.fliplr(img_a)
            img_b = np.fliplr(img_b)

    return img_a, img_b


def imread(path, is_gray_scale=False, img_size=None):
    if is_gray_scale:
        img = scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float)

        if not (img.ndim == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))

    if img_size is not None:
        img = scipy.misc.imresize(img, img_size)

    return img


def load_image(image_path, which_direction=0, is_gray_scale=True, img_size=(256, 256, 1)):
    input_img = imread(image_path, is_gray_scale=is_gray_scale, img_size=img_size)
    w_pair = int(input_img.shape[1])
    w_single = int(w_pair / 2)

    if which_direction == 0:    # A to B
        img_a = input_img[:, 0:w_single]
        img_b = input_img[:, w_single:w_pair]
    else:                       # B to A
        img_a = input_img[:, w_single:w_pair]
        img_b = input_img[:, 0:w_single]

    return img_a, img_b


def load_data(image_path, flip=True, is_test=False, which_direction=0, is_gray_scale=True, img_size=(256, 256, 1)):
    img_a, img_b = load_image(image_path=image_path, which_direction=which_direction,
                              is_gray_scale=is_gray_scale, img_size=img_size)

    img_a, img_b = preprocess_pair(img_a, img_b, flip=flip, is_test=is_test)
    img_a = transform(img_a)
    img_b = transform(img_b)

    # hope output should be [h, w, c]
    if (img_a.ndim == 2) and (img_b.ndim == 2):
        img_a = np.expand_dims(img_a, axis=2)
        img_b = np.expand_dims(img_b, axis=2)

    return img_a, img_b


