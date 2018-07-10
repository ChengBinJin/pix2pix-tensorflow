# ---------------------------------------------------------
# Tensorflow Vanilla GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf

from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('dataset', 'facades', 'dataset name for choice [facades|maps], default: facades')
tf.flags.DEFINE_integer('which_direction', 0, 'AtoB (0) or BtoA (1), default: AtoB 0')

tf.flags.DEFINE_integer('batch_size', 1, 'batch size for one feed forwrad, default: 1')
tf.flags.DEFINE_bool('is_train', False, 'training or inference mode, default: False')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')

tf.flags.DEFINE_integer('iters', 20, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 1, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 10, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 5, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_batch', 4, 'number of sampling images for check generator quality, default: 4')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        print('=='*10)
        print('Training mode!')
        print('=='*10)
        solver.train()
    else:
        print('=='*10)
        print('Test mode!')
        print('=='*10)
        solver.test()


if __name__ == '__main__':
    tf.app.run()
