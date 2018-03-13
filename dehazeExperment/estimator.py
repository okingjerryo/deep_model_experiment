import tensorflow as tf
from dehazeExperment import args
from dehazeExperment.util import conv2d_with_pad as conv2d_SP


# 主函数中调用estimator的接口
def get_estimator(feature_columns):
    return tf.estimator.Estimator(
        model_fn=stack_CAGAN_main,
        params={
            'feature': feature_columns
        }
    )


def stack_CAGAN_main():
    pass


def encoder(input_X, name='encoder', activate_fn=tf.nn.leaky_relu):
    '''
        encoder 模型建立:使用downsampling
    :param input_X: input feature
    :param name:
    :return: output
    '''

    # 128
    conv1 = conv2d_SP(input_X, 64, kernel_size=4, strides=2, padding=1, name=name + '_conv1')

    conv2 = activate_fn(conv1, name=name + '_conv2_act')
    # 64
    conv2 = conv2d_SP(conv2, 128, kernel_size=4, strides=2, padding=1, name=name + '_conv2')
    conv2 = tf.layers.batch_normalization(conv2, name=name + '_conv2_bn')

    conv3 = activate_fn(conv2, name=name + '_conv3_act')
    # 32
    conv3 = conv2d_SP(conv3, 256, kernel_size=4, strides=2, padding=1, name=name + '_conv3')
    conv3 = tf.layers.batch_normalization(conv3, name=name + '_conv3_bn')

    conv4 = activate_fn(conv3, name=name + '_conv4_act')
    # 16
    conv4 = conv2d_SP(conv4, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv4')
    conv4 = tf.layers.batch_normalization(conv4, name=name + '_conv4_bn')

    conv5 = activate_fn(conv4, name=name + '_conv5_act')
    # 8
    conv5 = conv2d_SP(conv5, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv5')
    conv5 = tf.layers.batch_normalization(conv5, name=name + '_conv5_bn')

    conv6 = activate_fn(conv5, name=name + '_conv6_act')
    # 4
    conv6 = conv2d_SP(conv6, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv6')
    conv6 = tf.layers.batch_normalization(conv6, name=name + '_conv6_bn')

    conv7 = activate_fn(conv6, name=name + '_conv7_act')
    # 2
    conv7 = conv2d_SP(conv7, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv7')

    # downsample 1x1

    return {
        'output_conv': conv7,
        'conv_detail': [conv1, conv2, conv3, conv4, conv5, conv6, conv7]
    }


# todo: UNet ED coder

if __name__ == '__main__':
    a = tf.ones([40, 256, 256, 3])
    encoder1 = encoder(a, name='testEncoder')

    init_op = tf.global_variables_initializer()
    with tf.Session(config=args.gpu_option()) as sess:
        sess.run(init_op)
        out_dic = sess.run(encoder1)
        for layer in out_dic['conv_detail']:
            print(layer.shape)
