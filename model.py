'''
sunkejia
GAN network
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim


def inst_norm(inputs, epsilon=1e-3, suffix=''):
    """
    Assuming TxHxWxC dimensions on the tensor, will normalize over
    the H,W dimensions. Use this before the activation layer.
    This function borrows from:
    http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    Note this is similar to batch_normalization, which normalizes each
    neuron by looking at its statistics over the batch.
    :param input_:
    input tensor of NHWC format
    """
    # Create scale + shift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    print(stat_shape)
    scale = tf.get_variable('INscale' + suffix,
                            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('INshift' + suffix,
                            initializer=tf.zeros(stat_shape[3]))

    # batch  nrom axes=[0,1,2] 出来的结果只有1 * C,而 instanse norm 结果为 B* C
    inst_means, inst_vars = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)

    # Normalization
    inputs_normed = (inputs - inst_means) / tf.sqrt(inst_vars + epsilon)

    # Perform trainable shift.
    output = scale * inputs_normed + shift
    return output


def dense_block(layer_input, kernel_size, k, filiter_num, scopename):
    # todo: desnet修改
    pass


def netG_encoder_gamma_32(image_input, reuse=False):
    '''
    08-04 删除了line reshape层
    :param image_input:
    :param reuse:
    :return:
    '''
    with tf.variable_scope('generator', reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size = [3, 3]
        filter_num = 32
        imageshape = image_input.get_shape().as_list()[1]
        print(imageshape)
        with tf.variable_scope('encoding'):
            # 目前用的是lrelu，其实应该用elu，后面注意跟换
            with slim.arg_scope([slim.conv2d], normalizer_fn=inst_norm, activation_fn=tf.nn.elu, padding='SAME',
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                # 32
                net = slim.conv2d(image_input, filter_num, kernel_size, normalizer_fn=None, scope='conv1')
                net = slim.conv2d(net, filter_num * 2, kernel_size, scope='conv2')
                # 16
                net = slim.conv2d(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='conv3')
                net = slim.conv2d(net, filter_num * 4, kernel_size, scope='conv4')
                # 8
                net = slim.conv2d(net, filter_num * 4, stride=2, kernel_size=kernel_size, scope='conv6')
                net = slim.conv2d(net, filter_num * 6, kernel_size, scope='conv7')
                # 4
                net = slim.conv2d(net, filter_num * 6, stride=2, kernel_size=kernel_size, scope='conv9')
                net = slim.conv2d(net, filter_num * 8, kernel_size, scope='conv10')

                net = tf.reshape(slim.flatten(net),
                                 [-1, 1, 1, int(imageshape / 8) * int(imageshape / 8) * filter_num * 8], name='fc1')
                logits = slim.fully_connected(net, 64, activation_fn=None, normalizer_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                              scope='bottleneck')
            output = logits  # 512维的向量
            return output


def netG_deconder_gamma_32(feature, output_channel, reuse=False):
    '''
    01-02 instanse norm
    @brief:
        feature:1*1*320+13+50
        pose:1*1*13r
        noise:1*1*50
    '''
    with tf.variable_scope('generator', reuse=reuse):
        kernel_size = [3, 3]
        filter_num = 32
        with tf.variable_scope('decoding') as vs:
            if reuse:
                vs.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.elu, normalizer_fn=inst_norm,
                                padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                # 先将vector组织为6*6*320的tensor#slim.batch_norm
                fc1 = slim.fully_connected(feature, 4 * 4 * filter_num * 8, activation_fn=None,
                                           weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           scope='fc1')
                # reshape the vector[n,6,6,320]
                inputs_img = tf.reshape(fc1, [-1, 4, 4, filter_num * 8])
                # print 'inputs_img',inputs_img.shape
                # 4
                net = slim.conv2d_transpose(inputs_img, filter_num * 8, kernel_size, scope='deconv01')
                net = slim.conv2d_transpose(net, filter_num * 6, kernel_size, scope='deconv02')
                # 8
                net = slim.conv2d_transpose(net, filter_num * 6, stride=2, kernel_size=kernel_size, scope='deconv2')
                net = slim.conv2d_transpose(net, filter_num * 4, kernel_size, scope='deconv3')
                # 16
                net = slim.conv2d_transpose(net, filter_num * 4, stride=2, kernel_size=kernel_size, scope='deconv5')
                net = slim.conv2d_transpose(net, filter_num * 2, kernel_size, scope='deconv6')
                # 32
                net = slim.conv2d_transpose(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='deconv8')
                net = slim.conv2d_transpose(net, filter_num, kernel_size, scope='deconv9')
            # 为什么放到外面就好了呢？
            net = slim.conv2d_transpose(net, output_channel, kernel_size, activation_fn=tf.nn.tanh, normalizer_fn=None,
                                        scope='deconv13', weights_initializer=tf.contrib.layers.xavier_initializer())
            output = net
        return output


def netD_discriminator_adloss_32(image_input, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as vs:
        kernel_size = [3, 3]
        filter_num = 32
        imageshape = image_input.get_shape().as_list()[1]
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.elu, padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            # 224/96/32
            net = slim.conv2d(image_input, filter_num, kernel_size, normalizer_fn=None, scope='conv1')
            net = slim.conv2d(net, filter_num * 2, kernel_size, scope='conv2')
            # 112/48/16
            net = slim.conv2d(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='conv3')
            net = slim.conv2d(net, filter_num * 2, kernel_size, scope='conv4')
            # 56/24/8
            net = slim.conv2d(net, filter_num * 4, stride=2, kernel_size=kernel_size, scope='conv6')
            net = slim.conv2d(net, filter_num * 4, kernel_size, scope='conv7')
            # 28/12/4
            net = slim.conv2d(net, filter_num * 6, stride=2, kernel_size=kernel_size, scope='conv9')
            net = slim.conv2d(net, filter_num * 6, kernel_size, scope='conv10')
            # 14/6/2
            net = slim.conv2d(net, filter_num * 8, stride=2, kernel_size=kernel_size, scope='conv12')
            net = slim.conv2d(net, filter_num * 8, kernel_size, scope='conv13')

            avgpool = slim.pool(net, [int(imageshape / 16), int(imageshape / 16)], stride=int(imageshape / 32),
                                pooling_type="AVG", scope='avgpool')
            adlogits = slim.fully_connected(slim.flatten(avgpool), 1, activation_fn=None, normalizer_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            scope='ad_soft')

            return adlogits


def netG_encoder_gamma(image_input, reuse=False):
    '''
    08-04 删除了line reshape层
    :param image_input:
    :param reuse:
    :return:
    '''
    with tf.variable_scope('generator', reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size = [3, 3]
        filter_num = 32
        imageshape = image_input.get_shape().as_list()[1]
        print(imageshape)
        with tf.variable_scope('encoding'):
            # 目前用的是lrelu，其实应该用elu，后面注意跟换
            with slim.arg_scope([slim.conv2d], normalizer_fn=inst_norm, activation_fn=tf.nn.elu, padding='SAME',
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                # 224
                net = slim.conv2d(image_input, filter_num, kernel_size, normalizer_fn=None, scope='conv1')
                net = slim.conv2d(net, filter_num * 2, kernel_size, scope='conv2')
                # 112
                net = slim.conv2d(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='conv3')
                net = slim.conv2d(net, filter_num * 2, kernel_size, scope='conv4')
                # 56
                net = slim.conv2d(net, filter_num * 4, stride=2, kernel_size=kernel_size, scope='conv6')
                net = slim.conv2d(net, filter_num * 3, kernel_size, scope='conv7')
                # 28
                net = slim.conv2d(net, filter_num * 6, stride=2, kernel_size=kernel_size, scope='conv9')
                net = slim.conv2d(net, filter_num * 4, kernel_size, scope='conv10')
                # 14
                net = slim.conv2d(net, filter_num * 8, stride=2, kernel_size=kernel_size, scope='conv12')
                net = slim.conv2d(net, filter_num * 6, kernel_size, scope='conv13')
                # avg出来之后应该是1*1*320的tensor
                # 7
                net = slim.conv2d(net, filter_num * 10, stride=2, kernel_size=kernel_size, scope='conv15')
                net = slim.conv2d(net, filter_num * 8, kernel_size, scope='conv16')

                net = tf.reshape(slim.flatten(net),
                                 [-1, 1, 1, int(imageshape / 32) * int(imageshape / 32) * filter_num * 8])
                logits = slim.fully_connected(net, 512, activation_fn=None, normalizer_fn=None,
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                              scope='bottleneck')
            output = logits  # 512维的向量
            return output


def netG_deconder_gamma(feature, output_channel, reuse=False):
    '''
    01-02 instanse norm
    @brief:
        feature:1*1*320+13+50
        pose:1*1*13r
        noise:1*1*50
    '''
    with tf.variable_scope('generator', reuse=reuse):
        kernel_size = [3, 3]
        filter_num = 32
        with tf.variable_scope('decoding') as vs:
            if reuse:
                vs.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.elu, normalizer_fn=inst_norm,
                                padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                # 先将vector组织为6*6*320的tensor#slim.batch_norm
                fc1 = slim.fully_connected(feature, 3 * 3 * filter_num * 8, activation_fn=None,
                                           weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           scope='fc1')
                # reshape the vector[n,6,6,320]
                inputs_img = tf.reshape(fc1, [-1, 3, 3, 256])
                # print 'inputs_img',inputs_img.shape
                # 7
                net = slim.conv2d(inputs_img, filter_num * 8, kernel_size, scope='deconv01')
                net = slim.conv2d(net, filter_num * 10, kernel_size, scope='deconv02')

                # 14
                net = slim.conv2d_transpose(net, filter_num * 10, stride=2, kernel_size=kernel_size, scope='deconv03')
                net = slim.conv2d_transpose(net, filter_num * 6, kernel_size, scope='deconv0')
                # 28
                net = slim.conv2d_transpose(net, filter_num * 8, stride=2, kernel_size=kernel_size, scope='deconv2')
                net = slim.conv2d_transpose(net, filter_num * 4, kernel_size, scope='deconv3')
                # 56
                net = slim.conv2d_transpose(net, filter_num * 6, stride=2, kernel_size=kernel_size, scope='deconv5')
                net = slim.conv2d_transpose(net, filter_num * 3, kernel_size, scope='deconv6')
                # 112
                net = slim.conv2d_transpose(net, filter_num * 4, stride=2, kernel_size=kernel_size, scope='deconv8')
                net = slim.conv2d_transpose(net, filter_num * 2, kernel_size, scope='deconv9')
                # 224
                net = slim.conv2d_transpose(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='deconv11')
                net = slim.conv2d_transpose(net, filter_num * 1, kernel_size, scope='deconv12')
            # 为什么放到外面就好了呢？
            net = slim.conv2d_transpose(net, output_channel, kernel_size, activation_fn=tf.nn.tanh, normalizer_fn=None,
                                        scope='deconv13', weights_initializer=tf.contrib.layers.xavier_initializer())
            output = net
        return output


def netD_discriminator_adloss(image_input, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as vs:
        kernel_size = [3, 3]
        filter_num = 32
        imageshape = image_input.get_shape().as_list()[1]
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.elu, padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(
                                stddev=0.02)):
            # 224/96/32
            net = slim.conv2d(image_input, filter_num, kernel_size, normalizer_fn=None, scope='conv1')
            net = slim.conv2d(net, filter_num * 2, kernel_size, scope='conv2')
            # 112/48/16
            net = slim.conv2d(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='conv3')
            net = slim.conv2d(net, filter_num * 2, kernel_size, scope='conv4')
            # 56/24/8
            net = slim.conv2d(net, filter_num * 4, stride=2, kernel_size=kernel_size, scope='conv6')
            net = slim.conv2d(net, filter_num * 4, kernel_size, scope='conv7')
            # 28/12/4
            net = slim.conv2d(net, filter_num * 6, stride=2, kernel_size=kernel_size, scope='conv9')
            net = slim.conv2d(net, filter_num * 6, kernel_size, scope='conv10')
            # 14/6/2
            net = slim.conv2d(net, filter_num * 8, stride=2, kernel_size=kernel_size, scope='conv12')
            net = slim.conv2d(net, filter_num * 8, kernel_size, scope='conv13')
            # two path -feature -W Omegapredict_r_label
            # avg出来之后应该是1*1*320的tensor
            # 7/3
            # net = slim.conv2d(net, filter_num * 10, stride=2, kernel_size=kernel_size, scope='conv15')
            net = slim.conv2d(net, filter_num * 10, kernel_size, scope='conv16')

            avgpool = slim.pool(net, [int(imageshape / 32), int(imageshape / 32)], stride=int(imageshape / 32),
                                pooling_type="AVG", scope='avgpool')
            adlogits = slim.fully_connected(slim.flatten(avgpool), 1, activation_fn=None, normalizer_fn=None,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            scope='ad_soft')

            return adlogits


def netG_Unet_decoder_gamma_32(feature, output_channel, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        kernel_size = [3, 3]
        filter_num = 32
        with tf.variable_scope('decoding') as vs:
            if reuse:
                vs.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.elu, normalizer_fn=inst_norm,
                                padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                # 先将vector组织为6*6*320的tensor#slim.batch_norm
                fc1 = slim.fully_connected(feature, 4 * 4 * filter_num * 8, activation_fn=None,
                                           weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           scope='fc1')

                # reshape the vector[n,6,6,320]
                inputs_img = tf.reshape(fc1, [-1, 4, 4, filter_num * 8])
                # Unet改变
                fc1_en = slim.get_variables_by_name('fc1', 'encoding')[0]
                tf.concat([fc1_en, inputs_img], axis=3, name='defc1')
                # print 'inputs_img',inputs_img.shape
                # 4
                net = slim.conv2d(inputs_img, filter_num * 8, kernel_size, scope='deconv01')
                net = slim.conv2d(net, filter_num * 6, kernel_size, scope='deconv02')
                # 8
                net = slim.conv2d_transpose(net, filter_num * 3, stride=2, kernel_size=kernel_size, scope='deconv2')
                net = slim.conv2d_transpose(net, filter_num * 4, kernel_size, scope='deconv3')
                # 16
                net = slim.conv2d_transpose(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='deconv5')
                net = slim.conv2d_transpose(net, filter_num * 2, kernel_size, scope='deconv6')
                # 32
                net = slim.conv2d_transpose(net, filter_num * 2, stride=2, kernel_size=kernel_size, scope='deconv8')
                net = slim.conv2d_transpose(net, filter_num, kernel_size, scope='deconv9')
            # 为什么放到外面就好了呢？
            net = slim.conv2d_transpose(net, output_channel, kernel_size, activation_fn=tf.nn.tanh, normalizer_fn=None,
                                        scope='deconv13', weights_initializer=tf.contrib.layers.xavier_initializer())
            output = net
        return output
