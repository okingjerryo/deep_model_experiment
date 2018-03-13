import time
from datetime import datetime
from dehazeExperment import args
import tensorflow as tf

def get_timestamp_now():
    '''
    获取当前的时间戳，为integer型
    :return: int型的时间戳
    '''
    thistime = int(time.mktime(datetime.now().timetuple()))
    return thistime


def conv2d_with_pad(input, filter, padding, kernel_size, strides, name='', activation=None):
    '''指定padding的conv2d,用于全卷积层downsampling
        pad 为一个Integer，定义了图像2,3维度pad的数量
    '''
    # 卷积
    conv = tf.layers.conv2d(input, filter, kernel_size, strides, name=name + '_conv')
    # padding
    pad_mat = [[0, 0], [padding, 0], [0, padding], [0, 0]]
    pad = tf.pad(conv, pad_mat, name=name + '_pad')
    return pad


if __name__ == '__main__':
    # test padding
    a = tf.ones([1, 256, 256, 3])

    a_c = conv2d_with_pad(a, 64, padding=1, kernel_size=[4, 4], strides=2, name='conv_1')

    a_relu = tf.nn.leaky_relu(a_c, name='conv_1_act')
    b_c = conv2d_with_pad(a_relu, 128, padding=1, kernel_size=4, strides=2, name='conv_2')
    b_norm = tf.layers.batch_normalization(b_c)

    init = tf.global_variables_initializer()
    with tf.Session(config=args.gpu_option()) as sess:
        sess.run(init)
        print(sess.run(a).shape)
        print(sess.run(a_relu).shape)
        print(sess.run(b_norm).shape)
