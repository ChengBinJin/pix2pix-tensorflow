# ---------------------------------------------------------
# Tensorflow Vanilla GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset import dataset
from pix2pix import Pix2Pix
# import tensorflow_utils as tf_utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.image_size = (256, 256, 3)
        self.dataset = dataset(self.flags.dataset, image_size=self.image_size)
        self.model = Pix2Pix(self.sess, self.flags, image_size=self.image_size)
        self._make_folders()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        # tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.model_out_dir = "{}_{}/model/{}".format(self.flags.dataset, self.flags.which_direction, cur_time)
            self.sample_out_dir = "{}_{}/sample/{}".format(self.flags.dataset, self.flags.which_direction, cur_time)
            self.train_writer = tf.summary.FileWriter('{}_{}/logs/{}'.format(self.flags.dataset,
                                                                             self.flags.which_direction, cur_time))

            if not os.path.isdir(self.model_out_dir):
                os.makedirs(self.model_out_dir)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

        elif not self.flags.is_train:
            self.model_out_dir = "{}_{}/model/{}".format(self.flags.dataset, self.flags.which_direction,
                                                         self.flags.load_model)
            self.test_out_dir = "{}_{}/test/{}".format(self.flags.dataset, self.flags.which_direction,
                                                       self.flags.load_model)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def train(self):
        for iter_time in range(self.flags.iters):
            # samppling images and save them
            self.sample(iter_time)

            # train_step
            imgs_x, imgs_y = self.dataset.train_next_batch(batch_size=self.flags.batch_size,
                                                           which_direction=self.flags.which_direction)
            loss, summary = self.model.train_step(imgs_x, imgs_y)
            self.model.print_info(loss, iter_time)
            self.train_writer.add_summary(summary, iter_time)
            self.train_writer.flush()

            # save model
            self.save_model(iter_time)

        self.save_model(self.flags.iters)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        num_iters = int(self.dataset.num_tests / self.flags.sample_batch)
        total_time = 0.
        for iter_time in range(num_iters):
            print(iter_time)

            # measure inference time
            start_time = time.time()
            imgs_x, imgs_y = self.dataset.val_next_batch(batch_size=self.flags.sample_batch,
                                                         which_direction=self.flags.which_direction, is_train=False)
            imgs = self.model.sample_imgs(imgs_x, imgs_y)  # inference
            total_time += time.time() - start_time
            self.model.plots(imgs, iter_time, self.test_out_dir)

        print('Avg PT: {:.2f} msec.'.format(total_time / num_iters * 1000.))

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            imgs_x, imgs_y = self.dataset.val_next_batch(batch_size=self.flags.sample_batch,
                                                         which_direction=self.flags.which_direction)
            imgs = self.model.sample_imgs(imgs_x, imgs_y)
            self.model.plots(imgs, iter_time, self.sample_out_dir)

    def save_model(self, iter_time):
        if np.mod(iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)

            print('=====================================')
            print('             Model saved!            ')
            print('=====================================\n')

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))
            return True
        else:
            return False
