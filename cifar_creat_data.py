from numpy import *
import os
import sys
import tensorflow as tf
import scipy.io as scio
import matplotlib.pyplot as plt

slim = tf.contrib.slim




def data_to_tfrecord(images, labels, filename):

    print('Converting data into %s ...' % filename)

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            writer = tf.python_io.TFRecordWriter(filename)
            image = tf.placeholder(tf.uint8, shape=[32, 32, 3])
            image_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        for index, img in enumerate(images):
            img_jpeg = sess.run(image_jpeg, feed_dict={image: img})
            label = int(labels[0, index])
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label])),
                        'image':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img_jpeg]))
                    }))
            writer.write(example.SerializeToString())
            if index % 100 == 0:
                print(index)

        writer.close()
    return


def write_data():
    if FLAGS.cifar10:
        data = scio.loadmat('all-cifar/cifar-10-batches-mat/cifar10.mat')
        print(data['train_image'].shape)
        print(data['train_label'].shape)
        data_to_tfrecord(data['train_image'], data['train_label'],
                         'all-cifar/cifar-10-batches-mat/cifar_train.tfrecord')
        data_to_tfrecord(data['test_image'], data['test_label'],
                         'all-cifar/cifar-10-batches-mat/cifar_test.tfrecord')
    else:
        data = scio.loadmat('cifar-100-python/cifar100.mat')
        print(data['train_image'].shape)
        print(data['train_label'].shape)
        data_to_tfrecord(data['train_image'], data['train_label'],
                         'cifar-100-python/cifar_train.tfrecord')
        data_to_tfrecord(data['test_image'], data['test_label'],
                         'cifar-100-python/cifar_test.tfrecord')


def read_and_decode(filename, batchsize=30, aug=['train'], img_mean=None):
    # with tf.device('/cpu:0'):
    print('read and decode data')
    filename_queue = tf.train.string_input_producer(filename, num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })

    img = tf.image.decode_jpeg(features['image'], channels=3)
    img = dataAug(img, aug=aug, img_mean=img_mean)
    label = tf.cast(features['label'], tf.int32)

    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label],
        batch_size=batchsize,
        capacity=1000,
        min_after_dequeue=500,
        num_threads=2)
    
    print('read and decode data done')

    return img_batch, label_batch


def dataAug(img, aug=['train'], img_mean=None):
    print('dataAug start')
    if 'cal_mean' not in aug:
        if img_mean is not None:
            img_mean = scio.loadmat(img_mean)
        else:
            if FLAGS.cifar10:
                img_mean = scio.loadmat(
                    'all-cifar/cifar-10-batches-mat/cifar_mean.mat')
                print('cifar10_dataAug')
            else:
                img_mean = scio.loadmat('cifar-100-python/cifar_mean.mat')
                print('cifar100_dataAug')
    if 'train' in aug:
        img = tf.image.resize_images(img, [32, 32], method=0)
        img = tf.cast(img, tf.float32)
        if 'no_sub_mean' not in aug:
            img = img - img_mean['cifar_mean']
        print(img.shape)
        img = tf.image.resize_image_with_crop_or_pad(img, 40, 40)
        img = tf.random_crop(img, [32, 32, 3])
        img = tf.image.random_flip_left_right(img)
    if 'test' in aug:
        img = tf.image.resize_images(img, [32, 32], method=0)
        img = tf.cast(img, tf.float32)
        if 'no_sub_mean' not in aug:
            img = img - img_mean['cifar_mean']

    if 'cal_mean' in aug:
        img = tf.image.resize_images(img, [32, 32], method=0)

    img = tf.cast(img, tf.float32)
    print('dataAug done')
    return img


def cal_mean():
    if FLAGS.cifar10:
        trainfile = ['all-cifar/cifar-10-batches-mat/cifar_train.tfrecord']
    else:
        trainfile = ['cifar-100-python/cifar_train.tfrecord']
    with tf.device('/cpu:0'):
        img_train, label_train = read_and_decode(
            trainfile, batchsize=500, aug=['cal_mean'])
        img_mean = tf.reduce_mean(img_train, axis=0)

    # saver=tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        img = sess.run(img_mean)
        print(img.shape)
        for i in range(99):
            _img = sess.run(img_mean)
            img += _img
            print(i)
            print(mean(img))
        img1 = img + 0.
        coord.request_stop()
        coord.join(threads)
    img1 /= 100.
    print(mean(img1))
    if FLAGS.cifar10:
        scio.savemat('all-cifar/cifar-10-batches-mat/cifar_mean.mat', {
            'cifar_mean': img1
        })
    else:
        scio.savemat('cifar-100-python/cifar_mean.mat', {'cifar_mean': img1})


def data_provid_test():
    if FLAGS.cifar10:
        trainfile = ['all-cifar/cifar-10-batches-mat/cifar_train.tfrecord']
        print('train file is %s' % trainfile)
    else:
        trainfile = ['cifar-100-python/cifar_train.tfrecord']
        print('train file is %s' % trainfile)

    with tf.device('/cpu:0'):
        img_train, label_train = read_and_decode(
            trainfile, batchsize=1, aug=['train', 'no_sub_mean'])

    # saver=tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        for i in range(100):
            img = sess.run(img_train)
            img = img/255.
            img.shape = [32, 32, 3]
            # img = img(0, ...)
            print(img.shape)
            plt.imshow(img)
            plt.show()

        coord.request_stop()


if __name__ == '__main__':
    tf.app.flags.DEFINE_boolean('cifar10', True, 'define if the data is cifar10')
    tf.app.flags.DEFINE_string('gpu', '2', 'set gpu')
    FLAGS = tf.app.flags.FLAGS

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # cal_mean()
    # data_provid_test()
    FLAGS.cifar10 = True
    # write_data()
    data_provid_test()
