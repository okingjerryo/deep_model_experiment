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


def conv2d_with_pad(input, filter, padding, kernel_size, strides, name='',reuse = False):
    '''指定padding的conv2d,用于全卷积层downsampling
        pad 为一个Integer，定义了图像2,3维度pad的数量
    '''
    # 卷积
    conv = tf.layers.conv2d(input, filter, kernel_size, strides, name=name + '_conv',reuse=reuse)
    # padding
    pad_mat = [[0, 0], [padding, 0], [0, padding], [0, 0]]
    pad = tf.pad(conv, pad_mat, name=name + '_pad')
    return pad

if __name__ == '__main__':
    pass
