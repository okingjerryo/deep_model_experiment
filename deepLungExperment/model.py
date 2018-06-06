from deepLungExperment import est_component as estimator
import tensorflow as tf
import collections  # 定义容器
from deepLungExperment.args import a
import deepLungExperment.args

Model = collections.namedtuple("Model",
                               "op,predict_real,predict_fake,discrim_loss,discrim_grads_and_vars,gen_loss_GAN,gen_loss_L1,gen_grads_and_vars,train")


def stack_CAGAN_main(input_X, input_Y):
    O_1 = estimator.unet_generator(input_X, name="O1_encoder")
    O_1_op = O_1.output_conv
    O_1_D = estimator.vgg19_descriminator(O_1_op, name="O1_discriminator")
    O_2 = estimator.unet_generator(O_1_op, name="O2_encoder")
    O_2_op = O_2.output_conv
    O_2_D = estimator.vgg19_descriminator(O_2_op, name="O2_discriminator")
    return Model(
        predict_real=tf.group(O_1_D.logist, O_2_D.logist),
        op=O_2_D
    )


def cGAN_main(input_X, input_Y):
    with tf.variable_scope("generator"):
        out_channels = int(input_Y.get_shape()[-1])
        outputs = estimator.cgan_unet_g(input_X, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = estimator.cgan_patch_d(input_X, input_Y)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = estimator.cgan_patch_d(input_X, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + args.EPS) + tf.log(1 - predict_fake + args.EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + args.EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(input_Y - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    # 滑动平均值
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        op=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train)
    )
