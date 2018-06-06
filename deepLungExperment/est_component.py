import tensorflow as tf
import collections
from deepLungExperment import args
from deepLungExperment.op import conv2d_with_pad as conv2d_SP
from deepLungExperment.op import *

# normalize interface
E_Component = collections.namedtuple("E_Component", "output_conv,conv_detail")


def unet_generator(input_X, name='encoder', conv_act_fn=tf.nn.leaky_relu, deconv_act_fun=tf.nn.relu):
    '''
        encoder 模型建立:使用downsampling
    :param input_X: input feature
    :param name:
    :return: output
    '''

    # 128
    conv1 = conv2d_SP(input_X, 64, kernel_size=4, strides=2, padding=1, name=name + '_conv1')

    conv2 = conv_act_fn(conv1, name=name + '_conv2_act')
    # 64
    conv2 = conv2d_SP(conv2, 128, kernel_size=4, strides=2, padding=1, name=name + '_conv2')
    conv2 = tf.layers.batch_normalization(conv2, name=name + '_conv2_bn')

    conv3 = conv_act_fn(conv2, name=name + '_conv3_act')
    # 32
    conv3 = conv2d_SP(conv3, 256, kernel_size=4, strides=2, padding=1, name=name + '_conv3')
    conv3 = tf.layers.batch_normalization(conv3, name=name + '_conv3_bn')

    conv4 = conv_act_fn(conv3, name=name + '_conv4_act')
    # 16
    conv4 = conv2d_SP(conv4, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv4')
    conv4 = tf.layers.batch_normalization(conv4, name=name + '_conv4_bn')

    conv5 = conv_act_fn(conv4, name=name + '_conv5_act')
    # 8
    conv5 = conv2d_SP(conv5, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv5')
    conv5 = tf.layers.batch_normalization(conv5, name=name + '_conv5_bn')

    conv6 = conv_act_fn(conv5, name=name + '_conv6_act')
    # 4
    conv6 = conv2d_SP(conv6, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv6')
    conv6 = tf.layers.batch_normalization(conv6, name=name + '_conv6_bn')

    conv7 = conv_act_fn(conv6, name=name + '_conv7_act')
    # 2
    conv7 = conv2d_SP(conv7, 512, kernel_size=4, strides=2, padding=1, name=name + '_conv7')

    # todo 3: downsample 1x1
    # upsample
    dconv7 = deconv_act_fun(conv7, name=name + '_dconv7_act')
    # 4
    dconv7 = tf.layers.conv2d_transpose(dconv7, 512, kernel_size=4, strides=2, padding='same', name=name + '_dconv7')
    dconv7 = tf.layers.batch_normalization(dconv7, name=name + '_dconv7_bn')
    # concat—— Unet
    dconv7 = tf.concat([dconv7, conv6], -1, name=name + '_dconv7_cat')

    dconv6 = deconv_act_fun(dconv7, name=name + '_dconv6_act')
    # 8
    dconv6 = tf.layers.conv2d_transpose(dconv6, 512, kernel_size=4, strides=2, padding='same', name=name + '_dconv6')
    dconv6 = tf.layers.batch_normalization(dconv6, name=name + '_dconv6_bn')
    dconv6 = tf.concat([dconv6, conv5], -1, name=name + '_dconv6_cat')

    dconv5 = deconv_act_fun(dconv6, name=name + '_dconv5_act')
    # 16 512
    dconv5 = tf.layers.conv2d_transpose(dconv5, 512, kernel_size=4, strides=2, padding='same', name=name + '_dconv5')
    dconv5 = tf.layers.batch_normalization(dconv5, name=name + '_dconv5_bn')
    dconv5 = tf.concat([dconv5, conv4], -1, name=name + '_dconv5_cat')

    dconv4 = deconv_act_fun(dconv5, name=name + '_dconv4_act')
    # 32 256
    dconv4 = tf.layers.conv2d_transpose(dconv4, 256, kernel_size=4, strides=2, padding='same', name=name + '_dconv4')
    dconv4 = tf.layers.batch_normalization(dconv4, name=name + '_dconv4_bn')
    dconv4 = tf.concat([dconv4, conv3], -1, name=name + '_dconv4_cat')

    dconv3 = deconv_act_fun(dconv4, name=name + '_dconv3_act')
    # 64 128
    dconv3 = tf.layers.conv2d_transpose(dconv3, 128, kernel_size=4, strides=2, padding='same', name=name + '_dconv3')
    dconv3 = tf.layers.batch_normalization(dconv3, 1, name=name + '_dconv3_bn')
    dconv3 = tf.concat([dconv3, conv2], -1, name=name + '_dconv3_cat')

    dconv2 = deconv_act_fun(dconv3, name=name + '_dconv2_act')
    # 128 64
    dconv2 = tf.layers.conv2d_transpose(dconv2, 64, kernel_size=4, strides=2, padding='same', name=name + '_dconv2')
    dconv2 = tf.layers.batch_normalization(dconv2, name=name + '_dconv2_bn')
    dconv2 = tf.concat([dconv2, conv1], -1, name=name + '_dconv2_cat')

    dconv1 = deconv_act_fun(dconv2, name=name + '_dconv1_act')
    # 256 1 x
    dconv1 = tf.layers.conv2d_transpose(dconv1, 3, kernel_size=4, strides=2, padding='same', name=name + '_dconv1')
    dconv1 = tf.nn.tanh(dconv1, name=name + 'final_act')

    return E_Component(
        output_conv=dconv1,
        conv_detail=[dconv7, dconv6, dconv5, dconv4, dconv3, dconv2, dconv1]
    )


# from cGAN
def cgan_unet_g(input_X, final_output_channals):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(input_X, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, final_output_channals)
        output = tf.tanh(output)
        layers.append(output)

    return E_Component(
        output_conv=layers[-1],
        conv_detail=layers
    )


# from cGAN
def cgan_patch_d(input_X, input_Y):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([input_X, input_Y], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = discrim_conv(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return E_Component(
        output_conv=layers[-1],
        conv_detail=layers
    )


if __name__ == '__main__':
    # a = tf.ones([10, 512, 512, 3])
    # dict = stack_CAGAN_main(a,a)
    # init_op = tf.global_variables_initializer()
    # with tf.Session(config=args.gpu_option()) as sess:
    #     sess.run(init_op)
    #     dic = sess.run(dict)
    #     print(dic)
    a = tf.placeholder(shape=[None, 4, 4, 128])
    sess = tf.Session(config=args.gpu_option())
