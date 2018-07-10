import random
import numpy as np
from datetime import datetime

import utils as utils


def dataset(dataset_name, image_size):
    if (dataset_name == 'facades') or (dataset_name == 'maps'):
        return FacadesMaps(dataset_name, image_size)
    else:
        raise NotImplementedError


class FacadesMaps(object):
    def __init__(self, dataset_name, image_size):
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.num_trains, self.num_vals, self.num_tests = 0, 0, 0
        self.test_index = 0

        self.train_data_path = '../../Data/{}/train'.format(self.dataset_name)
        self.val_data_path = '../../Data/{}/val'.format(self.dataset_name)
        self.test_data_path = '../../Data/{}/{}'.format(self.dataset_name,
                                                        'test' if self.dataset_name == 'facades' else 'val')

        self.is_gray = False
        self._load_data()

    def _load_data(self):
        print('Load {} dataset...'.format(self.dataset_name))

        self.train_data = utils.all_files_under(self.train_data_path, extension='.jpg')
        self.val_data = utils.all_files_under(self.val_data_path, extension='.jpg')
        self.test_data = utils.all_files_under(self.test_data_path, extension='.jpg')

        self.num_trains = len(self.train_data)
        self.num_vals = len(self.val_data)
        self.num_tests = len(self.test_data)

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size=1, which_direction=0):
        random.seed(datetime.now())  # set random seed
        batch_files = np.random.choice(self.train_data, batch_size, replace=False)

        data_x, data_y = [], []
        for batch_file in batch_files:
            batch_x, batch_y = utils.load_data(image_path=batch_file, which_direction=which_direction,
                                               is_gray_scale=self.is_gray, img_size=self.image_size)
            data_x.append(batch_x)
            data_y.append(batch_y)

        batch_ximgs = np.asarray(data_x).astype(np.float32)  # list to array
        batch_yimgs = np.asarray(data_y).astype(np.float32)  # list to array

        return batch_ximgs, batch_yimgs

    def val_next_batch(self, batch_size=1, which_direction=0, is_train=True):
        if is_train:
            random.seed(datetime.now())  # set random seed
            batch_files = np.random.choice(self.val_data, batch_size, replace=False)
        else:
            batch_files = self.test_data[self.test_index:self.test_index + batch_size]
            self.test_index += batch_size

        data_x, data_y = [], []
        for batch_file in batch_files:
            batch_x, batch_y = utils.load_data(image_path=batch_file, flip=False, is_test=True,
                                               which_direction=which_direction, is_gray_scale=self.is_gray,
                                               img_size=self.image_size)

            data_x.append(batch_x)
            data_y.append(batch_y)

        batch_ximg = np.asarray(data_x).astype(np.float32)  # list to array
        batch_yimg = np.asarray(data_y).astype(np.float32)  # list to array

        return batch_ximg, batch_yimg
