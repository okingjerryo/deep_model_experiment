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


if __name__ == '__main__':
    a = tf.ones([40, 256, 256, 3])
    generator1 = unet_generator(a, name='testGenerator')

    init_op = tf.global_variables_initializer()
    with tf.Session(config=args.gpu_option()) as sess:
        sess.run(init_op)
        out_dic = sess.run(generator1)
        for layer in out_dic['conv_detail']:
            print(layer.shape)
