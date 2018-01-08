'''
sunkejia
GAN network
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
from tensorflow.python.ops import math_ops
from tensorflow.contrib.layers.python.layers import initializers

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
    print (stat_shape)
    scale = tf.get_variable('INscale'+suffix,
            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('INshift'+suffix,
            initializer=tf.zeros(stat_shape[3]))

    #batch  nrom axes=[0,1,2] 出来的结果只有1 * C,而 instanse norm 结果为 B* C
    inst_means, inst_vars = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)

    # Normalization
    inputs_normed = (inputs - inst_means) / tf.sqrt(inst_vars + epsilon)

    # Perform trainable shift.
    output = scale * inputs_normed + shift
    return output

def netG_encoder_gamma(image_input,reuse=False):
    '''
    08-04 删除了line reshape层
    :param image_input:
    :param reuse:
    :return:
    '''
    with tf.variable_scope('generator',reuse=reuse) as vs:
        if reuse:
            vs.reuse_variables()
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('encoding'):
            #目前用的是lrelu，其实应该用elu，后面注意跟换
            with slim.arg_scope([slim.conv2d],normalizer_fn=inst_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):

                #224
                conv1 = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
                conv2 = slim.conv2d(conv1,filter_num*2,kernel_size,scope='conv2')
                conv2_1 = slim.conv2d(conv2, filter_num * 2, kernel_size, scope='conv2_1')
                #112
                conv3 = slim.conv2d(conv2,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
                conv4 = slim.conv2d(conv3,filter_num*2,kernel_size,scope='conv4')
                conv5 = slim.conv2d(conv4,filter_num*4,kernel_size,scope='conv5')
                #56
                conv6 = slim.conv2d(conv5,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
                conv7 = slim.conv2d(conv6,filter_num*3,kernel_size,scope='conv7')
                conv8 = slim.conv2d(conv7,filter_num*6,kernel_size,scope='conv8')
                #28
                conv9 = slim.conv2d(conv8,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
                conv10= slim.conv2d(conv9,filter_num*4,kernel_size,scope='conv10')
                conv11= slim.conv2d(conv10,filter_num*8,kernel_size,scope='conv11')
                #14
                conv12= slim.conv2d(conv11,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
                conv13 = slim.conv2d(conv12, filter_num * 6, kernel_size, scope='conv13')
                conv14 = slim.conv2d(conv13, filter_num * 10, kernel_size, scope='conv14')
                # two path -feature -W Omegapredict_r_label
                # avg出来之后应该是1*1*320的tensor
                # 7
                conv15 = slim.conv2d(conv14, filter_num * 10, stride=2, kernel_size=kernel_size, scope='conv15')
                conv16 = slim.conv2d(conv15, filter_num * 8, kernel_size, scope='conv16')
                conv17 = slim.conv2d(conv16, filter_num * 12, kernel_size, scope='conv17')

                conv17=tf.reshape(slim.flatten(conv17),[-1,1,1,7*7*filter_num * 12])
                logits = slim.fully_connected(conv17, 512, activation_fn=None, normalizer_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                scope='bottleneck')
            output=logits#512维的向量
            return output

def netG_deconder_gamma(feature,output_channel,reuse=False):
    '''
    01-02 instanse norm
    @brief:
        feature:1*1*320+13+50
        pose:1*1*13r
        noise:1*1*50
    '''
    with tf.variable_scope('generator',reuse=reuse):
        kernel_size=[3,3]
        filter_num=32
        with tf.variable_scope('decoding') as vs:
            if reuse:
                vs.reuse_variables()
            with slim.arg_scope([slim.conv2d_transpose],activation_fn=tf.nn.elu,normalizer_fn=inst_norm,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                #先将vector组织为6*6*320的tensor#slim.batch_norm
                fc1 = slim.fully_connected(feature,7*7*384,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='fc1')
                #reshape the vector[n,6,6,320]
                inputs_img=tf.reshape(fc1,[-1,7,7,384])
                # print 'inputs_img',inputs_img.shape
                #7
                deconv01 = slim.conv2d(inputs_img,filter_num * 8,kernel_size, scope='deconv01')
                deconv02 = slim.conv2d(deconv01, filter_num * 10, kernel_size, scope='deconv02')

                #14
                deconv03 = slim.conv2d_transpose(deconv02, filter_num * 10, stride=2, kernel_size=kernel_size,scope='deconv03')
                deconv0 = slim.conv2d_transpose(deconv03,filter_num*6,kernel_size,scope='deconv0')
                deconv1 = slim.conv2d_transpose(deconv0,filter_num*8,kernel_size,scope='deconv1')
                #28
                deconv2 = slim.conv2d_transpose(deconv1,filter_num*8,stride=2,kernel_size=kernel_size,scope='deconv2')
                deconv3 = slim.conv2d_transpose(deconv2,filter_num*4,kernel_size,scope='deconv3')
                deconv4 = slim.conv2d_transpose(deconv3,filter_num*6,kernel_size,scope='deconv4')
                #56
                deconv5 = slim.conv2d_transpose(deconv4,filter_num*6,stride=2,kernel_size=kernel_size,scope='deconv5')
                deconv6 = slim.conv2d_transpose(deconv5,filter_num*3,kernel_size,scope='deconv6')
                deconv7 = slim.conv2d_transpose(deconv6,filter_num*4,kernel_size,scope='deconv7')
                #112
                deconv8 = slim.conv2d_transpose(deconv7,filter_num*4,stride=2,kernel_size=kernel_size,scope='deconv8')
                deconv9 = slim.conv2d_transpose(deconv8,filter_num*2,kernel_size,scope='deconv9')
                deconv10= slim.conv2d_transpose(deconv9,filter_num*2,kernel_size,scope='deconv10')
                #224
                deconv11= slim.conv2d_transpose(deconv10,filter_num*2,stride=2,kernel_size=kernel_size,scope='deconv11')
                deconv12= slim.conv2d_transpose(deconv11,filter_num*1,kernel_size,scope='deconv12')
            #为什么放到外面就好了呢？
            deconv13= slim.conv2d_transpose(deconv12,output_channel,kernel_size,activation_fn=tf.nn.tanh,normalizer_fn=None,scope='deconv13',weights_initializer=tf.contrib.layers.xavier_initializer())
            output=deconv13
        return output


def netD_discriminator_adloss(image_input,reuse=False):
    with tf.variable_scope('discriminator/adloss',reuse=reuse) as vs:
        kernel_size=[3,3]
        filter_num=32
        with slim.arg_scope([slim.conv2d],normalizer_fn=slim.batch_norm,activation_fn=tf.nn.elu,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            #224
            net = slim.conv2d(image_input,filter_num,kernel_size,normalizer_fn=None,scope='conv1')
            net = slim.conv2d(net,filter_num*2,kernel_size,scope='conv2')
            #112
            net = slim.conv2d(net,filter_num*2,stride=2,kernel_size=kernel_size,scope='conv3')
            net = slim.conv2d(net,filter_num*2,kernel_size,scope='conv4')
            net = slim.conv2d(net,filter_num*4,kernel_size,scope='conv5')
            # 56
            net = slim.conv2d(net,filter_num*4,stride=2,kernel_size=kernel_size,scope='conv6')
            net = slim.conv2d(net,filter_num*3,kernel_size,scope='conv7')
            net = slim.conv2d(net,filter_num*6,kernel_size,scope='conv8')
            #28
            net = slim.conv2d(net,filter_num*6,stride=2,kernel_size=kernel_size,scope='conv9')
            net= slim.conv2d(net,filter_num*4,kernel_size,scope='conv10')
            net= slim.conv2d(net,filter_num*8,kernel_size,scope='conv11')
            #14
            net = slim.conv2d(net,filter_num*8,stride=2,kernel_size=kernel_size,scope='conv12')
            net = slim.conv2d(net,filter_num*6,kernel_size,scope='conv13')
            net = slim.conv2d(net,filter_num*10,kernel_size,scope='conv14')
            #two path -feature -W Omegapredict_r_label
            #avg出来之后应该是1*1*320的tensor
            #7
            net = slim.conv2d(net, filter_num * 10, stride=2, kernel_size=kernel_size, scope='conv15')
            net = slim.conv2d(net, filter_num * 8, kernel_size, scope='conv16')
            net = slim.conv2d(net, filter_num * 12, kernel_size, scope='conv17')

            avgpool=slim.pool(net,[7,7],stride=7,pooling_type="AVG",scope='avgpool')
            adlogits = slim.fully_connected(slim.flatten(avgpool),1,activation_fn=None,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),scope='ad_soft')

            return adlogits

