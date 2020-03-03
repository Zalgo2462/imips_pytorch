import os.path

import numpy as np
import tensorflow as tf

"""
run_imips_tf_loss.py runs the extracted loss routine from IMIPS (tensorflow) against
1000 test cases of synthetic IMIPS network output on 256 image patches each. It reads the test case numpy files
(scores.npy and labels.npy) from ./test_cases/test_case_####. If the test case data does not exist, the script
will create it. The imips loss and the sub-losses for each case are then saved to
./test_cases/test_case_####/losses.npy.  

This script is meant to be run against tensorflow v1.
"""

EPS = 10e-5


def inlierLoss(x):
    return -tf.log(x + EPS)


def outlierLoss(x):
    return -tf.log(tf.maximum(1 - x, EPS))


class CorrespTrainNet(object):
    def __init__(self):
        """
        patch_sz = FLAGS.depth * 2 + 1
        batch_sz = FLAGS.chan
        """
        patch_sz = 15
        batch_sz = 128

        # Patches and outputs
        for attr in ['ip', 'corr']:
            """
            setattr(self, attr + '_patches', tf.placeholder(
                tf.float32, [batch_sz, patch_sz, patch_sz, 1]))
            setattr(self, attr + '_outs', draft2Net(
                getattr(self, attr + '_patches'), reuse=tf.AUTO_REUSE))
            """
            setattr(self, attr + '_outs', tf.placeholder(
                tf.float32, [batch_sz, 1, 1, 128]))

        # Labels in 1D
        # Note that outlier is not simply 'not inlier', as we might leave it
        # open for some patches not to be trained either way.
        for attr in ['inlier', 'outlier']:
            setattr(self, 'is_' + attr + '_label', tf.placeholder(
                tf.bool, [batch_sz]))

        # Cast labels into proper shape: Things that happen on the
        # batch-channel diagonal
        for attr in ['inlier', 'outlier']:
            setattr(self, 'is_' + attr + '_diag', tf.expand_dims(tf.expand_dims(
                tf.matrix_diag(
                    getattr(self, 'is_' + attr + '_label')), 1), 1))

        # LOSSES
        # Pull up correspondences (outlier)
        self.own_corr_outs = tf.boolean_mask(
            self.corr_outs, self.is_outlier_diag)
        self.corresp_loss = tf.reduce_sum(inlierLoss(self.own_corr_outs))
        # Pull up inliers
        self.inlier_outs = tf.boolean_mask(
            self.ip_outs, self.is_inlier_diag)
        self.inlier_loss = tf.reduce_sum(inlierLoss(self.inlier_outs))
        # Push down outliers
        self.outlier_outs = tf.boolean_mask(
            self.ip_outs, self.is_outlier_diag)
        self.outlier_loss = tf.reduce_sum(outlierLoss(self.outlier_outs))
        # Suppress other channels at inliers:
        self.is_noninlier_activation = tf.logical_xor(
            self.is_inlier_diag, tf.expand_dims(tf.expand_dims(
                tf.expand_dims(self.is_inlier_label, -1), -1), -1))

        self.noninlier_activations = tf.boolean_mask(
            self.ip_outs, self.is_noninlier_activation)
        # TODO nonlinear robust loss?
        self.suppression_loss = tf.reduce_sum(self.noninlier_activations)

        self.loss = 1 * self.corresp_loss + self.outlier_loss + \
                    self.suppression_loss + self.inlier_loss
        """
        self.loss = FLAGS.corr_w * self.corresp_loss + self.outlier_loss + \
                    self.suppression_loss + self.inlier_loss
        """

        """
        self.train_step = tf.train.AdamOptimizer(
            10 ** -FLAGS.lr).minimize(self.loss)
        """


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

for i in range(1000):
    test_case_dir = "./test_case_" + str(i) + "/"

    if os.path.exists(test_case_dir):
        print
        "Loading data"
        scores = np.load(test_case_dir + "scores.npy")
        labels = np.load(test_case_dir + "labels.npy")
    else:
        print
        "Creating data"
        scores = np.random.rand(256, 1, 1, 128)
        labels = np.random.rand(256) > .5
        try:
            os.mkdir(test_case_dir)
        except:
            pass
        np.save(test_case_dir + "scores.npy", scores)
        np.save(test_case_dir + "labels.npy", labels)

    ip_outs = scores[0:scores.shape[0] // 2]
    corr_outs = scores[scores.shape[0] // 2:scores.shape[0]]
    inlier_labels = labels[0:labels.shape[0] // 2]
    outlier_labels = labels[labels.shape[0] // 2:labels.shape[0]]

    sess = tf.Session()
    loss_obj = CorrespTrainNet()

    outputs = sess.run(
        [loss_obj.loss, loss_obj.corresp_loss, loss_obj.inlier_loss, loss_obj.outlier_loss, loss_obj.suppression_loss],
        feed_dict={
            loss_obj.ip_outs: ip_outs,
            loss_obj.corr_outs: corr_outs,
            loss_obj.is_inlier_label: inlier_labels,
            loss_obj.is_outlier_label: outlier_labels,
        })

    losses = np.array(outputs)
    np.save(test_case_dir + "losses.npy", losses)
