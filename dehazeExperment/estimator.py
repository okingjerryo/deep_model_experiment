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


def stack_CAGAN_main(input_X):
    O_1 = unet_generator(input_X, name="O1_encoder")
    O_1_op = O_1["output_conv"]
    O_1_D = vgg19_descriminator(O_1_op, name="O1_discriminator")
    O_2 = unet_generator(O_1_op, name="O2_encoder")
    O_2_op = O_2["output_conv"]
    O_2_D = vgg19_descriminator(O_2_op, name="O2_discriminator")
    return {
        "D_detail": [O_1_D["logist"], O_2_D["logist"]],
        "op": O_2
    }


def unet_generator(input_X, name='encoder', conv_act_fn=tf.nn.leaky_relu,deconv_act_fun = tf.nn.relu):
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
    dconv7 = deconv_act_fun(conv7,name= name+'_dconv7_act')
    # 4
    dconv7 = tf.layers.conv2d_transpose(dconv7,512,kernel_size=4,strides=2,padding='same',name= name+'_dconv7')
    dconv7 = tf.layers.batch_normalization(dconv7,name= name+'_dconv7_bn')
    # concat—— Unet
    dconv7 = tf.concat([dconv7,conv6],-1,name= name+'_dconv7_cat')

    dconv6 = deconv_act_fun(dconv7,name= name+'_dconv6_act')
    # 8
    dconv6 = tf.layers.conv2d_transpose(dconv6,512,kernel_size=4,strides=2,padding='same',name= name+'_dconv6')
    dconv6 = tf.layers.batch_normalization(dconv6,name= name+'_dconv6_bn')
    dconv6 = tf.concat([dconv6,conv5],-1,name= name+'_dconv6_cat')

    dconv5 = deconv_act_fun(dconv6,name= name+'_dconv5_act')
    # 16 512
    dconv5 = tf.layers.conv2d_transpose(dconv5,512,kernel_size=4,strides=2,padding='same',name= name+'_dconv5')
    dconv5 = tf.layers.batch_normalization(dconv5, name=name + '_dconv5_bn')
    dconv5 = tf.concat([dconv5, conv4], -1, name=name + '_dconv5_cat')

    dconv4 = deconv_act_fun(dconv5, name=name + '_dconv4_act')
    # 32 256
    dconv4 = tf.layers.conv2d_transpose(dconv4,256,kernel_size=4,strides=2,padding='same',name=name + '_dconv4')
    dconv4 = tf.layers.batch_normalization(dconv4,name=name + '_dconv4_bn')
    dconv4 = tf.concat([dconv4,conv3],-1,name=name + '_dconv4_cat')

    dconv3 = deconv_act_fun(dconv4,name=name + '_dconv3_act')
    # 64 128
    dconv3 = tf.layers.conv2d_transpose(dconv3,128,kernel_size=4,strides=2,padding='same',name=name + '_dconv3')
    dconv3 = tf.layers.batch_normalization(dconv3,1,name=name + '_dconv3_bn')
    dconv3 = tf.concat([dconv3,conv2],-1,name=name + '_dconv3_cat')

    dconv2 = deconv_act_fun(dconv3,name=name + '_dconv2_act')
    # 128 64
    dconv2 = tf.layers.conv2d_transpose(dconv2,64,kernel_size=4,strides=2,padding='same',name=name + '_dconv2')
    dconv2 = tf.layers.batch_normalization(dconv2,name=name + '_dconv2_bn')
    dconv2 = tf.concat([dconv2,conv1],-1,name=name + '_dconv2_cat')

    dconv1 = deconv_act_fun(dconv2,name=name + '_dconv1_act')
    # 256 1 x
    dconv1 = tf.layers.conv2d_transpose(dconv1,3,kernel_size=4,strides=2,padding='same',name=name + '_dconv1')
    dconv1 = tf.nn.tanh(dconv1,name=name+'final_act')

    return {
        'output_conv': dconv1,
        'conv_detail': [dconv7, dconv6, dconv5, dconv4, dconv3, dconv2, dconv1]
    }


def vgg19_descriminator(input_X, name='descriminator', conv_act_fn=tf.nn.leaky_relu):
    # 128
    conv1 = tf.layers.conv2d(input_X, 64, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv1_1')
    conv1 = tf.layers.conv2d(conv1, 64, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv1_2')
    conv1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, name=name + '_conv1_Mpool')

    # 64
    conv2 = tf.layers.conv2d(conv1, 128, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv2_1')
    conv2 = tf.layers.conv2d(conv2, 128, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv2_2')
    conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, name=name + '_conv2_pool')

    # 32
    conv3 = tf.layers.conv2d(conv2, 256, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv3_1')
    conv3 = tf.layers.conv2d(conv3, 256, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv3_2')
    conv3 = tf.layers.conv2d(conv3, 256, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv3_3')
    conv3 = tf.layers.conv2d(conv3, 256, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv3_4')
    conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, name=name + '_conv3_Mpool')

    # 16
    conv4 = tf.layers.conv2d(conv3, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv4_1')
    conv4 = tf.layers.conv2d(conv4, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv4_2')
    conv4 = tf.layers.conv2d(conv4, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv4_3')
    conv4 = tf.layers.conv2d(conv4, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv4_4')
    conv4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2, name=name + '_conv4_Mpool')

    # 8
    conv5 = tf.layers.conv2d(conv4, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv5_1')
    conv5 = tf.layers.conv2d(conv5, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv5_2')
    conv5 = tf.layers.conv2d(conv5, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv5_3')
    conv5 = tf.layers.conv2d(conv5, 512, kernel_size=3, padding='same', activation=conv_act_fn, name=name + '_conv5_4')
    conv5 = tf.layers.max_pooling2d(conv5, pool_size=2, strides=2, name=name + '_conv5_Mpool')

    # float reshape
    f_conv5 = tf.reshape(conv5, [conv5.shape[0], conv5.shape[1] * conv5.shape[2] * conv5.shape[3]],
                         name=name + "_conv5_flatten")
    # FC
    fc_1 = tf.layers.dense(f_conv5, 4096, activation=conv_act_fn, name=name + "_fc1")
    fc_1 = tf.layers.dropout(fc_1, name=name + "_fc1_dp")
    fc_2 = tf.layers.dense(fc_1, 1000, activation=conv_act_fn, name=name + "_fc2")
    fc_2 = tf.layers.dropout(fc_2, name=name + "_fc2_dp")
    # soft_max
    logist = tf.layers.dense(fc_2, 1, activation=tf.nn.softmax)
    return {"logist": logist,
            "conv_detail": [conv1, conv2, conv3, conv4, conv5, f_conv5, fc_1, fc_2]
            }
if __name__ == '__main__':
    a = tf.ones([128, 256, 256, 3])
    dict = stack_CAGAN_main(a)
    init_op = tf.global_variables_initializer()
    with tf.Session(config=args.gpu_option()) as sess:
        sess.run(init_op)
        dic = sess.run(dict["op"])
        print(dic["output_conv"].shape)
