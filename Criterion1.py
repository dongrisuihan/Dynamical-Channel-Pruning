import os
import sys
import gc
import time
import logging

import scipy.io as scio
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim
from numpy import *
import numpy as np
import numpy.random as nr
import tensorflow.examples.tutorials.mnist.input_data as input_data
sys.path.append('../../../data/cifar')
import cifar_creat_data
from cifar_creat_data import read_and_decode
from visdom import Visdom

tf.app.flags.DEFINE_string('gpu', '2', 'set gpu')

tf.app.flags.DEFINE_string(
    'rootdir',
    './savefile/cifar100/accPrune/res44_2_prune0.5_long2',
    'set root directory to save file')

tf.app.flags.DEFINE_boolean('restore', False, 'set whether restore from file')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
logger = logging.getLogger(__name__)

class Cifar_ResNet:
    def __init__(self):
        # parameter
        logger.info('\n' + '*' * 100 + '\n' + '******init******\n' + '*' * 100)
        self.cifar10 = False
        self.rate = 1
        self.batchsize = 120
        self.withdropout = True
        self.withprune = True
        self.loadprune = False
        self.loadpruneshuffle = False
        self.n = {
            'block1': 3,
            'block2': 3,
            'block3': 3,
        }
        if self.loadprune:
            self.prune_ = scio.loadmat(FLAGS.rootdir + '/modelFile' + '/prune')
            FLAGS.rootdir = FLAGS.rootdir + '_load'
            if self.loadpruneshuffle:
                FLAGS.rootdir = FLAGS.rootdir + '_shuffle'
        self.dp = {}
        self.lr = {}
        if self.loadprune:
            self.max_iter = 224000
            self.dp['decay'] = [0, 64000, 96000]
            self.dp['rate'] = [0.05, 0.05, 0.01]
            self.lr['ini'] = 0.1
            self.lr['decay'] = [64000, 128000, 192000]
            self.lr['rate'] = [0.01, 0.001, 0.0001]
        else:
            self.max_iter = 436000
            self.dp['decay'] = [0, 96000, 226000]
            self.dp['rate'] = [0.05, 0.05, 0.01]
            self.lr['ini'] = 0.1
            self.lr['decay'] = [96000, 328000, 372000]
            self.lr['rate'] = [0.01, 0.001, 0.0001]
        self.prune_rate = 1
        self.dropout_prob = self.dp['rate'][0]
        self.test_iter = 4000
        self.acc_batchsize = 200
        self.acc_batch_num = 50
        self.snapshot = 64000
        self.momentum = 0.9
        self.weight_decay = 0.0002
        self.loss10_weight = 1.
        self.staircase = True
        self.bn = True
        self.gs = 0.
        self.bn_decay = 0.999
        logger.info(FLAGS.rootdir)
        logger.info('dropout:')
        logger.info(self.withdropout)
        logger.info('prune:')
        logger.info(self.withprune)
        self.summary_file = FLAGS.rootdir + '/summaryFile'
        self.model_file = FLAGS.rootdir + '/modelFile'
        self.snapshot_file = self.model_file + '/snapshot'
        self.datadir = '../../../data/cifar'
        if self.cifar10:
            self.trainfile = [
                self.datadir +
                '/all-cifar/cifar-10-batches-mat/cifar_train.tfrecord'
            ]
            self.testfile = [
                self.datadir +
                '/all-cifar/cifar-10-batches-mat/cifar_test.tfrecord'
            ]
            self.img_mean = self.datadir + '/all-cifar/cifar-10-batches-mat/cifar_mean.mat'
            logger.info('cifar10_init')
        else:
            self.trainfile = [
                self.datadir + '/cifar-100-python/cifar_train.tfrecord'
            ]
            self.testfile = [
                self.datadir + '/cifar-100-python/cifar_test.tfrecord'
            ]
            self.img_mean = self.datadir + '/cifar-100-python/cifar_mean.mat'
            logger.info('cifar100_init')

        # model structure
        if self.cifar10:
            self.num_classes = 10
        else:
            self.num_classes = 100
        self.block = ['block1', 'block2', 'block3']
        # rate = 2
        self.chnl = {
            'block1': int(16 * self.rate),
            'block2': int(32 * self.rate),
            'block3': int(64 * self.rate),
        }

        self.ep = {}
        self.creat_prune()
        self.allcomp = self.calcomp()
        self.viz = Visdom()

        self.data_provider()
        self.model()
        # self.loss()
        self.add_summary()

        # build session
        self.sess = tf.Session()
        if self.loadprune:
            self.loadPrune()
        logger.info('\n' + '*' * 100 + '\n' + '****init done****\n' +
                    '*' * 100)

    def creat_prune(self):
        logger.info('--------------Creat Pruning--------------')
        self.prune = {}
        self.prune['cd'] = {}
        self.prune['cp'] = {}
        self.prune['ca'] = {}
        self.prune['cdo'] = {}
        self.prune['cpr'] = {}
        self.prune['cdltE'] = {}
        self.prune['ld'] = {}
        self.prune['lp'] = {}
        self.prune['la'] = {}
        self.prune['ldo'] = {}
        self.prune['ldltE'] = {}
        self.prune['E'] = 0.
        self.p_ema = 0.9997
        self.p = {}
        self.p['n'] = 14
        self.p['decay'] = [96000 + i * 10000 for i in range(self.p['n'])]
        self.p['rate'] = [0.5**((i+1)/14) for i in range(self.p['n'])]
        # self.p['rate'] = [28] * self.p['n']
        logger.info(self.p['decay'])
        logger.info(self.p['rate'])

        self.p_gate = 0.1
        self.sci_p = 1.
        self.sci_c = 1.
        self.dropout_p = 1.
        self.dropout_c = 1.
        n = self.n['block1']
        self.prune['ld'] = np.ones(n * 3, dtype=np.float32)
        self.prune['lp'] = np.ones(n * 3, dtype=np.float32)
        self.prune['la'] = np.zeros(n * 3, dtype=np.float32)
        self.prune['ldo'] = np.ones(n * 3, dtype=np.float32)
        self.prune['ldltE'] = np.zeros(n * 3, dtype=np.float32)

        def rate(a):
            return self.dropout_c**(2. - a) * self.dropout_p**(a)

        a = (1. + 2. + 4.) / (rate(0.) + 2. * rate(1.) + 4. * rate(2.))

        for blk, name in enumerate(self.block):
            n = self.n[name]
            chnl = self.chnl[name]
            self.prune['cd'][name] = np.ones((n, 1, 1, chnl), dtype=np.float32)
            self.prune['cp'][name] = np.ones((n, 1, 1, chnl), dtype=np.float32)
            self.prune['cpr'][name] = np.ones(n, dtype=np.float32)
            self.prune['cdo'][name] = np.ones(
                (n, 1, 1, chnl), dtype=np.float32) * rate(blk) * a
            self.prune['ca'][name] = np.zeros(
                (n, 1, 1, chnl), dtype=np.float32)
            self.prune['cdltE'][name] = np.ones(
                (n, 1, 1, chnl), dtype=np.float32)

    def loadPrune(self):
        logger.info('--------------Load Pruning--------------')
        for name in self.block:
            self.prune['cp'][name] = self.prune_['cp'][name][0, 0]
            # print(np.shape(self.prune['cp'][name]))
            if self.loadpruneshuffle:
                np.random.shuffle(self.prune['cp'][name])
                logger.info('shuffle')
                for i in range(np.shape(self.prune['cp'][name])[0]):
                    np.random.shuffle(self.prune['cp'][name][i, 0, 0, :])
            logger.info(np.sum(self.prune['cp'][name], axis=3))

    def data_provider(self):
        logger.info('--------------Data Provider--------------')
        with tf.variable_scope('data'):
            logger.info('train : train data')
            self.img_train, label_train = read_and_decode(
                self.trainfile,
                self.batchsize,
                aug=['train'],
                img_mean=self.img_mean)
            self.label_train = tf.one_hot(label_train, self.num_classes)
            logger.info('test : test data')
            self.img_test, label_test = read_and_decode(
                self.testfile,
                self.acc_batchsize,
                aug=['test'],
                img_mean=self.img_mean)
            self.label_test = tf.one_hot(label_test, self.num_classes)
            logger.info('test : train data')
            self.img_test_train, label_test_train = read_and_decode(
                self.trainfile,
                self.acc_batchsize,
                aug=['test'],
                img_mean=self.img_mean)
            self.label_test_train = tf.one_hot(label_test_train,
                                               self.num_classes)

            self.img = tf.placeholder(tf.float32, [None, 32, 32, 3])
            self.label = tf.placeholder(tf.int32, [None, self.num_classes])

        return

    def model(self):
        logger.info('--------------Build model--------------')
        ep = self.ep
        is_training = tf.Variable(True, trainable=False, name='is_training')
        set_training = tf.assign(is_training, True, name='set_is_training')
        clear_training = tf.assign(
            is_training, False, name='clear_is_training')
        ep['is_training'] = is_training
        ep['set_training'] = set_training
        ep['clear_training'] = clear_training

        def average(losses):
            loss = []
            for l in losses:
                expanded_l = tf.expand_dims(l, 0)
                loss.append(expanded_l)
            loss = tf.concat(loss, 0)
            loss = tf.reduce_mean(loss, 0)
            return loss

        def average_gradients(tower_grads):
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                grads = []
                for g, _ in grad_and_vars:
                    logger.debug(g)
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(grads, 0)
                grad = tf.reduce_mean(grad, 0)
                grad_and_var = (grad, grad_and_vars[0][1])
                average_grads.append(grad_and_var)
            return average_grads

        def cal_l2():
            train_var = tf.trainable_variables()
            loss_l2 = tf.Variable(0., trainable=False)
            for var in train_var:
                if 'weights' in var.name:
                    logger.info(var.name)
                    loss_l2 = loss_l2 + tf.nn.l2_loss(var)
            loss_l2 = loss_l2 * self.weight_decay
            return loss_l2

        def netbody(img, reuse=False):
            with tf.variable_scope('ResNet', reuse=reuse):
                net = img
                with slim.arg_scope(
                    [slim.conv2d],
                        padding='SAME',
                        kernel_size=[3, 3],
                        activation_fn=tf.nn.relu,
                        weights_initializer=slim.variance_scaling_initializer(),
                        normalizer_fn=self.BN if self.bn else None,
                        normalizer_params={
                            'is_training': is_training,
                            'decay': self.bn_decay,
                            'reuse': reuse
                        } if self.bn else None):
                    net = slim.conv2d(net, self.chnl['block1'], scope='conv1')
                    shortcut = net
                    # ep['conv1'] = net
                    for blk, name in enumerate(self.block):
                        n = self.n[name]
                        chnl = self.chnl[name]

                        with tf.variable_scope(name):
                            self.prune[name] = tf.Variable(
                                np.ones((n, 1, 1, chnl), dtype=np.float32),
                                trainable=False,
                                name='prune')
                            self.prune['ph' + name] = tf.placeholder(
                                tf.float32, shape=[n, 1, 1, chnl])
                            self.prune['asn' + name] = tf.assign(
                                self.prune[name], self.prune['ph' + name])
                            prune = tf.split(self.prune[self.block[blk]], n)
                            logger.info(name)
                            for i in range(n):
                                with tf.variable_scope('unit' + str(i), reuse=reuse):
                                    if blk != 0 and i == 0:
                                        # no additional paras and computations shortcut
                                        shortcut = tf.nn.avg_pool(
                                            shortcut, [1, 2, 2, 1], [1, 2, 2, 1],
                                            'SAME')
                                        shortcut = tf.concat(
                                            [shortcut, shortcut * 0.], 3)
                                        net = shortcut * prune[i]
                                        net = slim.conv2d(net,
                                                        int(chnl / self.rate))
                                    else:
                                        net = net * prune[i]
                                        net = slim.conv2d(net,
                                                        int(chnl / self.rate))

                                    net = slim.conv2d(
                                        net, chnl, activation_fn=None)
                                    net = net * prune[i]
                                    shortcut = shortcut + net
                                    shortcut = tf.nn.relu(shortcut)
                                    net = shortcut
                net = tf.reduce_mean(
                    shortcut, [1, 2], keep_dims=False, name='pool')
                # ep['pool'] = net
                logit = slim.fully_connected(
                    net,
                    self.num_classes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='fc')
                return logit

        def calloss(logits, label):
            label = tf.cast(label, tf.float32)
            loss10 = self.loss10_weight * tf.losses.softmax_cross_entropy(
                label, logits)

            predict = tf.equal(tf.arg_max(logits, 1), tf.arg_max(label, 1))
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
            return loss10, accuracy


        global_step = tf.Variable(0, trainable=False, name='golbal_step')
        learning_rate = tf.Variable(self.lr['ini'], trainable=False)
        self.lr['ph'] = tf.placeholder(tf.float32)
        self.lr['asn'] = tf.assign(learning_rate, self.lr['ph'])
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, self.momentum, use_nesterov=True)

        gpu = FLAGS.gpu
        gpu = gpu.split(',')
        img = tf.split(self.img, len(gpu), 0)
        label = tf.split(self.label, len(gpu), 0)
        grads = []
        loss10 = []
        loss = []
        accuracy = []
        with tf.variable_scope('resnet'):
            for i in range(len(gpu)):
                with tf.device('/gpu:%d' % i):
                    logger.info('/gpu:%d' % i)
                    with tf.name_scope("tower_%d" % i):
                        with tf.name_scope("body%d" % i):
                            if i == 0:
                                logits = netbody(img[i])
                            else:
                                logits = netbody(img[i], reuse=True)
                        with tf.name_scope("loss%d" % i):
                            if i == 0:
                                loss_l2 = cal_l2()
                                train_var = tf.trainable_variables()
                            _loss10, _accuracy = calloss(logits, label[i])
                            _loss = _loss10 + loss_l2
                            grad_var = optimizer.compute_gradients(_loss, var_list=train_var)

                        loss10.append(_loss10)
                        loss.append(_loss)
                        accuracy.append(_accuracy)
                        grads.append(grad_var)
                        logger.info(len(grads))
            with tf.device('/cpu:0'):
                logger.info('/cpu:0')
                logger.info(len(grads))
                avg_grads = average_gradients(grads)
                logger.info(len(avg_grads))
                train_step = optimizer.apply_gradients(
                    avg_grads, global_step=global_step)
        ep['loss10'] = average(loss10)
        ep['loss'] = average(loss)
        ep['loss_l2'] = loss_l2
        ep['accuracy'] = average(accuracy)
        ep['logits'] = logits
        ep['learning_rate'] = learning_rate
        ep['global_step'] = global_step
        ep['train_step'] = train_step
        return

    def BN(self,
           input,
           scope='BatchNorm',
           is_training=True,
           decay=0.997,
           epsilon=1e-3,
           reuse=False):
        ep = self.ep
        with tf.name_scope(scope):
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                shape = input.get_shape().as_list()
                # print(shape)
                if reuse:
                    scale = tf.get_variable(name='scale')
                    shift = tf.get_variable(name='shift')
                    ema_mean = tf.get_variable(name='ema_mean')
                    ema_var = tf.get_variable(name='ema_var')
                else:
                    scale = tf.get_variable(
                        name='scale',
                        shape=shape[-1],
                        initializer=tf.constant_initializer(1.))
                    shift = tf.get_variable(
                        name='shift',
                        shape=shape[-1],
                        initializer=tf.constant_initializer(0.))
                    ema_mean = tf.get_variable(
                        name='ema_mean',
                        shape=shape[-1],
                        initializer=tf.constant_initializer(0.),
                        trainable=False)
                    ema_var = tf.get_variable(
                        name='ema_var',
                        shape=shape[-1],
                        initializer=tf.constant_initializer(1.),
                        trainable=False)
                mean, var = tf.nn.moments(input, [0, 1, 2])
                ep[tf.get_variable_scope().name + '/mean'] = mean
                ep[tf.get_variable_scope().name + '/var'] = var
                ep[ema_mean.name] = ema_mean
                ep[ema_var.name] = ema_var
                a = tf.cast(is_training, tf.float32)
                e_m = mean * (1. - decay) + decay * ema_mean
                e_v = var * (1. - decay) + decay * ema_var
                e_m = e_m * a + ema_mean * (1. - a)
                e_v = e_v * a + ema_var * (1. - a)
                a_mean = tf.assign(ema_mean, e_m)
                a_var = tf.assign(ema_var, e_v)
                # summary_bn_mean = tf.summary.scalar(
                #     'mean', tf.reduce_mean(mean))
                # summary_bn_var = tf.summary.scalar(
                #     'var', tf.reduce_mean(var))
                # summary_bn_ema_mean = tf.summary.scalar(
                #     'ema_mean',
                #     tf.reduce_mean(ema_mean))
                # summary_bn_ema_var = tf.summary.scalar(
                #     'ema_var',
                #     tf.reduce_mean(ema_var))
                # tf.add_to_collection(name='test', value=summary_bn_mean)
                # tf.add_to_collection(name='test', value=summary_bn_var)
                # tf.add_to_collection(name='test', value=summary_bn_ema_mean)
                # tf.add_to_collection(name='test', value=summary_bn_ema_var)

            with tf.control_dependencies([a_mean, a_var]):
                m, v = tf.cond(is_training, lambda: (mean, var),
                            lambda: (ema_mean, ema_var))
                output = tf.nn.batch_normalization(input, m, v, shift, scale,
                                                epsilon)
        return output

    def loss(self):
        label = tf.cast(self.label, tf.float32)
        logits = self.ep['logits']
        ep = self.ep

        with tf.name_scope('loss'):
            prob = tf.nn.softmax(logits)
            ep['prob'] = prob
            loss10 = self.loss10_weight * tf.losses.softmax_cross_entropy(
                label, logits)
            ep['loss10'] = loss10

            train_var = tf.trainable_variables()
            loss_l2 = tf.Variable(0., trainable=False)
            for var in train_var:
                if 'weights' in var.name:
                    logger.info(var.name)
                    loss_l2 = loss_l2 + tf.nn.l2_loss(var)
            loss_l2 = loss_l2 * self.weight_decay
            ep['loss_l2'] = loss_l2

            loss = loss10 + loss_l2
            ep['loss'] = loss

        with tf.name_scope('accuracy'):
            predict = tf.equal(tf.arg_max(logits, 1), tf.arg_max(label, 1))
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
            ep['predict'] = predict
            ep['accuracy'] = accuracy

        global_step = tf.Variable(0, trainable=False, name='golbal_step')
        # learning_rate = self.learning_rate
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate,
        #     global_step=global_step,
        #     decay_steps=self.decay_step,
        #     decay_rate=0.1,
        #     staircase=self.staircase)
        # learning_rate = tf.minimum(self.max_learning_rate, learning_rate)
        # learning_rate = tf.maximum(self.min_learning_rate, learning_rate)
        learning_rate = tf.Variable(self.lr['ini'], trainable=False)
        self.lr['ph'] = tf.placeholder(tf.float32)
        self.lr['asn'] = tf.assign(learning_rate, self.lr['ph'])

        train_opt = tf.train.MomentumOptimizer(
            learning_rate, self.momentum, use_nesterov=True)
        # train_opt=tf.train.AdamOptimizer(learning_rate)
        train_step = train_opt.minimize(
            loss,
            var_list=train_var,
            name="train_step",
            global_step=global_step)
        ep['learning_rate'] = learning_rate
        ep['global_step'] = global_step
        ep['train_step'] = train_step

        return

    def add_summary(self):
        logger.info('Add summary')
        ep = self.ep

        summary_loss10 = tf.summary.scalar('loss10', ep['loss10'])
        ep['summary_loss10'] = summary_loss10
        summary_loss_l2 = tf.summary.scalar('loss_l2', ep['loss_l2'])
        ep['summary_loss_l2'] = summary_loss_l2
        summary_loss = tf.summary.scalar('loss', ep['loss'])
        ep['summary_loss'] = summary_loss
        tf.add_to_collection(name='loss', value=summary_loss10)
        tf.add_to_collection(name='loss', value=summary_loss_l2)
        tf.add_to_collection(name='loss', value=summary_loss)
        summary_learning_rate = tf.summary.scalar('learning_rate',
                                                  ep['learning_rate'])
        tf.add_to_collection(name='loss', value=summary_learning_rate)
        ep['summary_learning_rate'] = summary_learning_rate

        reset_all = tf.Variable(0., trainable=False, name='reset_all')
        with tf.name_scope('train_and_test_accuracy'):
            ep, reset_all = self.add_test_summary(
                name='train_accuracy',
                reset_all=reset_all,
                acc=ep['accuracy'],
                end_points=ep,
                collect_name='train')
            ep, reset_all = self.add_test_summary(
                name='test_accuracy',
                reset_all=reset_all,
                acc=ep['accuracy'],
                end_points=ep,
                collect_name='test')
        with tf.name_scope('train_and_test_loss10'):
            ep, reset_all = self.add_test_summary(
                name='train_loss10',
                reset_all=reset_all,
                acc=ep['loss10'],
                end_points=ep,
                collect_name='train')
            ep, reset_all = self.add_test_summary(
                name='test_loss10',
                reset_all=reset_all,
                acc=ep['loss10'],
                end_points=ep,
                collect_name='test')
        with tf.name_scope('train_and_test_loss_l2'):
            ep, reset_all = self.add_test_summary(
                name='train_loss_l2',
                reset_all=reset_all,
                acc=ep['loss_l2'],
                end_points=ep,
                collect_name='train')
            ep, reset_all = self.add_test_summary(
                name='test_loss_l2',
                reset_all=reset_all,
                acc=ep['loss_l2'],
                end_points=ep,
                collect_name='test')
        with tf.name_scope('train_and_test_loss'):
            ep, reset_all = self.add_test_summary(
                name='train_loss',
                reset_all=reset_all,
                acc=ep['loss'],
                end_points=ep,
                collect_name='train')
            ep, reset_all = self.add_test_summary(
                name='test_loss',
                reset_all=reset_all,
                acc=ep['loss'],
                end_points=ep,
                collect_name='test')

        ep['reset_all'] = reset_all

    def add_test_summary(self,
                         name='train_accuracy',
                         reset_all=None,
                         acc=None,
                         end_points=None,
                         collect_name='train'):
        train_accuracy = tf.Variable(0., trainable=False, name=name)
        reset_train_accuracy = tf.assign(train_accuracy, 0)
        train_accuracy = tf.assign_add(train_accuracy,
                                       acc / float(self.acc_batch_num))
        end_points[name] = train_accuracy
        summary_train_accuracy = tf.summary.scalar(name, train_accuracy)
        tf.add_to_collection(name=collect_name, value=summary_train_accuracy)
        end_points['summary_' + name] = summary_train_accuracy
        reset_all = reset_all + reset_train_accuracy

        return end_points, reset_all

    def train(self):
        logger.debug('Training')
        ep = self.ep
        sess = self.sess
        # sess.run(ep['set_dropout'])
        sess.run(ep['set_training'])
        if self.withdropout:
            logger.debug('Drop out')
            self.drop()
        _img, _label = sess.run([self.img_train, self.label_train])
        _, mer, gs, acc = sess.run(
            [
                ep['train_step'], self.merged_loss, ep['global_step'],
                ep['accuracy']
            ],
            feed_dict={
                self.img: _img,
                self.label: _label
            })
        self.summary_writer.add_summary(mer, gs)
        self.gs = gs
        for i, a in enumerate(self.lr['decay']):
            if self.gs == a:
                sess.run(
                    self.lr['asn'],
                    feed_dict={self.lr['ph']: self.lr['rate'][i]})
        if self.withprune:
            self.update_acc(acc)
            self.pruning(gs)
        # print(gs)

        return

    def drop(self):
        for i, a in enumerate(self.dp['decay']):
            if self.gs == a:
                for name in self.block:
                    if i != 0:
                        cdo = self.prune['cdo'][name]
                        cdo[:] = cdo[:] / self.dp['rate'][i -
                                                          1] * self.dp['rate'][i]
                    else:
                        cdo = self.prune['cdo'][name]
                        cdo[:] = cdo[:] * self.dp['rate'][i]

        for blk, name in enumerate(self.block):
            n = self.n[name]
            chnl = self.chnl[name]
            cd = self.prune['cd'][name]
            cp = self.prune['cp'][name]
            cdo = self.prune['cdo'][name]
            s = np.shape(cd)
            cd[:] = cp[:]
            r = np.random.rand(*s)
            cd[r <= cdo] = 0.
            self.sess.run(
                self.prune['asn' + name],
                feed_dict={self.prune['ph' + name]: cd})
        return

    def prune_test(self):
        for name in self.block:
            cp = self.prune['cp'][name]
            self.sess.run(self.prune['asn' + name],
                          {self.prune['ph' + name]: cp})
            # print('prune test:%s:%f' % (name, np.sum(cp)))
        return

    def update_acc(self, acc):
        self.prune[
            'E'] = self.prune['E'] * self.p_ema + acc * (1. - self.p_ema)
        for blk, name in enumerate(self.block):
            cd = self.prune['cd'][name]
            cp = self.prune['cp'][name]
            ca = self.prune['ca'][name]
            cdo = self.prune['cdo'][name]
            dltE = self.prune['cdltE'][name]
            # ca[cp == 0.] = 1.
            m = (cd != 0.)
            ca[m] = ca[m] * self.p_ema + acc * (1. - self.p_ema)
            dltE[:] = (ca[:] - self.prune['E']) * (1. - cdo[:]) / cdo[:]
            # dltE[m] = (dltE[m] - np.mean(dltE[m]))*rate(blk)+np.mean(dltE[m])
        return

    def calcomp(self):
        a = [4., 2., 1.]
        compute = 0.
        for blk, name in enumerate(self.block):
            cp = self.prune['cp'][name]
            compute += np.sum(cp) * a[blk]
        return compute

    def pruning(self, gs):
        logger.debug('Pruning')
        def rate(a):
            return self.sci_c**(2. - a) * self.sci_p**(a)

        for i, a in enumerate(self.p['decay']):
            if self.gs == a:
                scio.savemat(self.model_file + '/prunebf' + str(i), self.prune)
                # for _ in range(self.p['rate'][i]):
                d1 = self.prune['cdltE']['block1']
                d2 = self.prune['cdltE']['block2']
                d3 = self.prune['cdltE']['block3']
                cp1 = self.prune['cp']['block1']
                cp2 = self.prune['cp']['block2']
                cp3 = self.prune['cp']['block3']
                d2 = np.append(d1[cp1 != 0], d2[cp2 != 0])
                d3 = np.append(d2, d3[cp3 != 0])
                me = np.mean(d3)
                while self.calcomp() / self.allcomp > self.p['rate'][i]:
                    a = 2.
                    for blk, name in enumerate(self.block):
                        cpr = self.prune['cpr'][name]
                        cp = self.prune['cp'][name]
                        dltE = self.prune['cdltE'][name]
                        m = (cp != 0.)
                        dltE[cp == 0.] = 0.
                        # print(np.shape(m))
                        # dltE[m] = (dltE[m] - np.mean(
                        #     dltE[m])) * rate(blk) + np.mean(dltE[m])
                        dltE[m] = (dltE[m] - me) * rate(blk) + me
                        # print(np.shape(dltE[m]))
                        d = dltE[cpr > self.p_gate, ...]
                        if len(d) > 0:
                            c = np.min(d)
                            # print('%s: c = %f' % (name, c))
                            a = np.min([a, c])
                            # print('%s: a = %f' % (name, a))
                            # print(c)
                    for name in self.block:
                        b = self.prune['cp'][name]
                        c = self.prune['cdltE'][name]
                        b[np.abs(c - a) < 1e-8] = 0.
                    for name in self.block:
                        a = np.sum(
                            self.prune['cp'][name], axis=3) / self.chnl[name]
                        a = np.reshape(a, (-1))
                        self.prune['cpr'][name][:] = a[:]
                scio.savemat(self.model_file + '/pruneaf' + str(i), self.prune)

                cd = self.prune['cd']
                cp = self.prune['cp']
                cdltE = self.prune['cdltE']
                cpr = self.prune['cpr']
                self.viz.text(
                    "<p style='color:red'>Block1<br>Prune every layer: {}<br>Prune rate: {}<br>Prune scale rate: {}</p><br>"
                    "<p style='color:blue'>Block2<br>Prune every layer: {}<br>Prune rate: {}<br>Prune scale rate: {}</p><br>"
                    "<p style='color:BlueViolet'>Block3<br>Prune every layer: {}<br>Prune rate: {}<br>Prune scale rate: {}</p><br>".
                    format(
                        np.sum(cp['block1'], axis=3),
                        cpr['block1'],
                        np.mean(cpr['block1']),
                        np.sum(cp['block2'], axis=3),
                        cpr['block2'],
                        np.mean(cpr['block2']),
                        np.sum(cp['block3'], axis=3),
                        cpr['block3'],
                        np.mean(cpr['block3']),
                    ),
                    win='cp')
                self.viz.text(
                    "<p style='color:red'>Block1<br>Drop: num: {:.4f}; all{}<br>Prune: num: {:.4f}; all:{}</p><br>"
                    "<p style='color:blue'>Block2<br>Drop: num: {:.4f}; all{}<br>Prune: num: {:.4f}; all:{}</p><br>"
                    "<p style='color:BlueViolet'>Block3<br>Drop: num: {:.4f}; all{}<br>Prune: num: {:.4f}; all:{}</p><br>".
                    format(
                        np.sum(cd['block1']),
                        cd['block1'], np.sum(cp['block1']), cp['block1'],
                        np.sum(cd['block2']),
                        cd['block2'], np.sum(cp['block2']), cp['block2'],
                        np.sum(cd['block3']), cd['block3'], np.sum(
                            cp['block3']), cp['block3']),
                    win='cd')
                self.viz.text(
                    "<p style='color:red'>Block1<br>ca: num: {:.4f}; all{}</p><br>"
                    "<p style='color:blue'>Block2<br>ca: num: {:.4f}; all{}</p><br>"
                    "<p style='color:BlueViolet'>Block3<br>ca: num: {:.4f}; all{}</p><br>".
                    format(
                        np.sum(cdltE['block1']), cdltE['block1'],
                        np.sum(cdltE['block2']), cdltE['block2'],
                        np.sum(cdltE['block3']), cdltE['block3']),
                    win='ca')

    def test(self):
        logger.debug('Test')
        ep = self.ep
        sess = self.sess
        # sess.run(ep['clear_dropout'])
        sess.run(ep['clear_training'])
        # train_accuracy
        sess.run(ep['reset_all'])
        self.prune_test()
        for i in range(self.acc_batch_num):
            _img, _label = sess.run(
                [self.img_test_train, self.label_test_train])
            train_acc, mer, gs = sess.run(
                [ep['train_accuracy'], self.merged_train, ep['global_step']],
                feed_dict={
                    self.img: _img,
                    self.label: _label
                })
        self.summary_writer.add_summary(mer, gs)

        # testn_accuracy
        sess.run(ep['reset_all'])
        for i in range(self.acc_batch_num):
            _img, _label = sess.run([self.img_test, self.label_test])
            test_acc, mer, self.gs = sess.run(
                [ep['test_accuracy'], self.merged_test, ep['global_step']],
                feed_dict={
                    self.img: _img,
                    self.label: _label
                })
        self.summary_writer.add_summary(mer, gs)
        logger.info('step: %d, Accuracy: train: %f, test: %f' % (gs, train_acc,
                                                                 test_acc))

        return

    def run_whole(self):
        logger.info('\n' + '*' * 100 + '\n' + '****Start Training****\n' +
                    '*' * 100)
        saver = tf.train.Saver()
        self.cal_distribute()
        with self.sess as sess:
            self.merged_loss = tf.summary.merge_all('loss')
            self.merged_train = tf.summary.merge_all('train')
            self.merged_test = tf.summary.merge_all('test')
            self.summary_writer = tf.summary.FileWriter(
                self.summary_file, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            if FLAGS.restore:
                saver.restore(sess, self.snapshot_file)
            else:
                saver.save(sess, self.snapshot_file)
            for j in range(self.max_iter):
                # saver.restore(sess,self.model_file)
                # train
                self.train()
                if (j + 1) % self.snapshot == 0:
                    saver.save(sess, self.snapshot_file, global_step=self.gs)
                    # saver.save(sess, self.snapshot_file)
                # test
                if (j + 1) % self.test_iter == 0:
                    self.test()

            saver.save(sess, self.snapshot_file, global_step=self.gs)
            coord.request_stop()
            coord.join(threads)
            sess.close()
        scio.savemat(self.model_file + '/prune', self.prune)
        self.cal_distribute()
        # logger.info(self.prune['cpr'])
        logger.info('\n' + '*' * 100 + '\n' + '****Training Done****\n' +
                    '*' * 100)

    def cal_distribute(self):
        n = self.n['block1']
        cpr = self.prune['cpr']
        cp = self.prune['cp']
        logger.info('channel after pruning in each layer')
        for blk in self.block:
            logger.info(np.sum(cp[blk], axis=3))
        al = []
        logger.info('channel prune rate')
        for blk in self.block:
            logger.info(blk)
            logger.info(cpr[blk])
        for blk in self.block:
            al.append(np.mean(cpr[blk])*self.rate)
        pass
        logger.info(al)
        a = al[0]
        b = al[1]
        c = al[2]
        flop = 32 * 32 * 3 * 3 * 16 * self.rate * 3
        flop += 32 * 32 * 3 * 3 * 16 * 16 * n * 2 * a
        flop += 16 * 16 * 3 * 3 * 32 * 32 * n * 2 * b - 16 * 16 * 3 * 3 * 16 * 32 * b
        flop += 8 * 8 * 3 * 3 * 64 * 64 * n * 2 * c - 8 * 8 * 3 * 3 * 64 * 32 * c

        para = 3*3*16*self.rate*3
        para += 3 * 3 * 16 * 16 * n * 2 * a
        para += 3 * 3 * 32 * 32 * n * 2 * b - 3 * 3 * 32 * 16 * b
        para += 3 * 3 * 64 * 64 * n * 2 * c - 3 * 3 * 64 * 32 * c

        a = 1.
        b = 1.
        c = 1.
        flop1 = 32 * 32 * 3 * 3 * 16 * self.rate * 3
        flop1 += 32 * 32 * 3 * 3 * 16 * 16 * n * 2 * a
        flop1 += 16 * 16 * 3 * 3 * 32 * 32 * n * 2 * b - 16 * 16 * 3 * 3 * 16 * 32 * b
        flop1 += 8 * 8 * 3 * 3 * 64 * 64 * n * 2 * c - 8 * 8 * 3 * 3 * 64 * 32 * c

        para1 = 3 * 3 * 16 * self.rate * 3
        para1 += 3 * 3 * 16 * 16 * n * 2 * a
        para1 += 3 * 3 * 32 * 32 * n * 2 * b - 3 * 3 * 32 * 16 * b
        para1 += 3 * 3 * 64 * 64 * n * 2 * c - 3 * 3 * 64 * 32 * c
        flop /= 1e6
        flop1 /= 1e6
        flopr = 1.-flop/flop1
        para /= 1e6
        para1 /= 1e6
        parar = 1.-para/para1

        logger.info(
            '\nFLOPS: after prune {:.2f}, original {:.2f}, pruned {:.2%}\nParas: after prune {:.2f}, original {:.2f}, pruned {:.2%}'.
            format(para, para1, parar, flop, flop1, flopr))


def logset():
    logger.debug('Logger set')
    logger.setLevel(level=logging.INFO)

    path = os.path.dirname(FLAGS.rootdir)
    print('dirname: ' + path)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        
    handler = logging.FileHandler(FLAGS.rootdir+'_logger.txt')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return


if __name__ == '__main__':
    logset()
    a = Cifar_ResNet()
    a.run_whole()
