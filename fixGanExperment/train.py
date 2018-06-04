#-*- coding: utf-8 -*-
'''
sunkejia
train file
python main.py
'''
import logging
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import trange

import model as nets
import op as util


class Trainer(object):
    def __init__(self, sample_interval=20, restored=False, batch_size=64,
                 epoch=240, log_dir='', d_learning_rate=0.0002, g_learning_rate=0.0002, beta1=0.5,
                 test_batch_size=128, model_name='', output_channel=3,
                 input_size=224, input_channel=3, data_loader_train=None,
                 gpus_list='', check_point='check_point', output_size=224,
                 version='', savepath='', imagepath='', logfile='',
                 summary_dir='', discribe='', data_loader_valid=None, noise_z=10):

        self.ifsave = True
        self.restored=restored
        self.d_lr=d_learning_rate
        self.g_lr=g_learning_rate
        self.beta1=beta1
        self.batch_size = batch_size
        self.test_batch_size=test_batch_size
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_size = output_size
        self.output_channel = output_channel
        self.sample_interval = sample_interval
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.epoch = epoch
        self.noise_z=noise_z

        #dir
        self.log_dir = log_dir
        self.savename=savepath
        self.result_path = imagepath
        #存储checkpoint文件路径
        self.check_point=check_point
        self.check_point_path=os.path.join(os.path.dirname(logfile),'checkpoint')
        self.logfile=logfile
        self.summarypath=summary_dir

        self.gpus_list=gpus_list
        self.model_name=model_name

        #save loss and vars
        self.g_loss=None
        self.d_loss=None
        self.varsg=None
        self.varsd=None
        self.version=version
        self._mkdir_result(self.log_dir)#save mode dir
        self._mkdir_result(self.summarypath)#save summary
        self._mkdir_result(self.savename)#save result dir
        self._mkdir_result(self.result_path)#save image dir
        self._mkdir_result(self.check_point_path)#checkpoint dir
        #存储Log文件
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',datefmt='%a,%d %b %Y %H:%M:%S',filename=self.logfile,filemode='w')
        logging.info('discribe{}'.format(discribe))
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        self.gpus_arr=np.asarray(self.gpus_list.split(','),np.int32)
        print('use gpu nums:',len(self.gpus_arr))
        self.gpus_count=len(self.gpus_arr)
        #是否多核训练
        if self.gpus_count>1:
            self.multigpus=True
        else:
            self.multigpus=False
        #统计计算多少个batch/epoch
        self.batch_idxs = self.data_loader_train.epoch_batch / self.gpus_count

        #初始化模型
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus_list

        self._init_model()
        # self._init_validation_model()
        #控制显存使用
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        #初始化模型参数
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=20)

        #加载预训练模型
        if self.restored:
            saver_init = tf.train.Saver(self.init_vars)
            saver_init.restore(self.sess,self.check_point)
        #存储tensor board需要的数据
        self.summary_write = tf.summary.FileWriter(self.summarypath + '/' + self.version+'_'+self.gpus_list, self.sess.graph)

        self.data_loader_valid.Dual_enqueueStart() # 开启测试队列
        self.data_loader_train.Dual_enqueueStart()#开启训练对别

    def _mkdir_result(self,str_dir):
        if not os.path.exists(str_dir):
            try:
                os.mkdir(str_dir)
            except:
                os.mkdir("./{}".format(str_dir))

    def _init_model(self):
        '''
        init modle for train
        :return:
        '''
        self.global_step = slim.get_or_create_global_step()
        self.batch_data = tf.placeholder(dtype=tf.float32,shape=[None,self.input_size,self.input_size,self.input_channel],name='input_images')#image

        #网络过程
        self._predict_gan()
        #损失公式
        self._loss_gan()
        #loss计算
        self._loss_compute()

        self.summary_train = tf.summary.merge_all()
        #获取训练的参数部分
        train_vars = tf.trainable_variables()
        self.varsg = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.varsd = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.init_vars=self.varsg+self.varsd
        #优化函数
        self._get_train_op(self.global_step)

    def train(self):
        '''
        train
        :return:
        '''
        start_time = time.time()

        curr_interval=0
        for epoch_n in trange(self.epoch): #trange
            for interval_i in range(int(self.batch_idxs)):
                batch_image=np.zeros([self.batch_size*self.gpus_count,self.input_size,self.input_size,self.input_channel],np.float32)

                for b_i in range(self.gpus_count):
                    batch_image[b_i*self.batch_size:(b_i+1)*self.batch_size,:,:,:]=self.data_loader_train.read_data_batch() # 删掉了 label
                #D
                _ ,loss_d=self.sess.run([self.train_d_op,self.d_loss],
                                        feed_dict={self.batch_data: batch_image})

                # G
                _ = self.sess.run(self.train_g_op,
                                  feed_dict={self.batch_data: batch_image})
                loss_g, train_summary, step \
                    = self.sess.run(
                    [self.g_loss, self.summary_train, self.global_step],
                    feed_dict={self.batch_data: batch_image})
                if interval_i % 25 == 0:
                    self.summary_write.add_summary(train_summary, global_step=step)

                    logging.info('Epoch [%4d/%4d] [gpu%s] [global_step:%d]time:%.2f h, d_loss:%.4f, g_loss:%.4f' \
                                 % (
                                 epoch_n, self.epoch, self.gpus_list, step, (time.time() - start_time) / 3600.0, loss_d,
                                 loss_g))

                if (curr_interval) % int(self.sample_interval * self.batch_idxs) == 0:
                    if self.ifsave and curr_interval != 0:
                        self.saver.save(self.sess,
                                   os.path.join(self.check_point_path, self.model_name),
                                   global_step=step)
                curr_interval+=1
            # 目的让网络训练到后面 L1权重越低
    def _get_train_op(self,global_step):
        '''
        梯度计算
        :param global_step: 迭代次数
        :return:
        '''
        optimizer_d = tf.train.AdamOptimizer(learning_rate=self.d_lr,beta1=self.beta1,name='optimizer_d')
        grads_and_var_d = optimizer_d.compute_gradients(self.d_loss,self.varsd,colocate_gradients_with_ops = True)
        grads_d,vars_d = zip(*grads_and_var_d)
        grads_d,_ =tf.clip_by_global_norm(grads_d,0.1)
        self.train_d_op = optimizer_d.apply_gradients(zip(grads_d,vars_d),global_step)

        optimizer_g = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=self.beta1,name='optimizer_g')
        grads_and_var_g = optimizer_g.compute_gradients(self.g_loss,self.varsg,colocate_gradients_with_ops = True)
        grads_g ,var_g = zip(*grads_and_var_g)
        grads_g , _ = tf.clip_by_global_norm(grads_g,0.1)
        self.train_g_op = optimizer_g.apply_gradients(zip(grads_g,var_g))
        return global_step


    def _predict_gan(self, reuse=False):
        '''
        网络训练
        :param reuse: True | False
        :return:
        '''
        self.logits = nets.netG_encoder_gamma_32(self.batch_data, reuse=reuse)
        self.noise = tf.random_uniform(shape=(self.batch_size, 1, 1, self.noise_z), minval=-1, maxval=1,
                                       dtype=tf.float32, name='input_noise')

        LogitsWithNoise = tf.concat([self.logits, self.noise], axis=3)
        self.output_syn = nets.netG_deconder_gamma_32(LogitsWithNoise, self.output_channel, reuse=reuse)
        self.data_gt, self.data_noise = tf.split(self.batch_data, 2, axis=0)
        # 给内容loss用，无论噪声图还是原图 生成gan 必须要贴近真实
        self.data_gt_total = tf.concat([self.data_gt, self.data_gt], axis=0)
        # cgan 方案 对于d而言 使用了 noise+真图作为真;output+noise作为假
        self.data_real_total = tf.concat([self.data_noise, self.data_gt], axis=0)
        self.data_fake_total = tf.concat([self.data_noise, self.output_syn], axis=0)

        self.logits_d_real = nets.netD_discriminator_adloss_32(self.data_real_total, reuse=tf.AUTO_REUSE)
        # self.logits_d_fake = nets.netD_discriminator_adloss_32(self.output_syn, reuse=True)
        self.logits_d_fake = nets.netD_discriminator_adloss_32(self.data_fake_total, reuse=True)
    def _loss_gan(self):
        '''
        loss 计算
        :return:
        '''
        with tf.name_scope('D_loss'):
            #adversarial
            self.ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.logits_d_real), logits=self.logits_d_real
            ))
            self.ad_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.logits_d_fake),logits=self.logits_d_fake
            ))
            # self.d_l2_regular = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope='discriminator'))*args.weight_L2_regular

        with tf.name_scope('G_loss'):
            self.ad_loss_syn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.logits_d_fake), logits=self.logits_d_fake,
            ))
            # 与原图的MSE
            self.constraint_loss = tf.reduce_mean(
                tf.losses.mean_squared_error(labels=self.data_gt_total, predictions=self.output_syn))
        with tf.name_scope('loss'):
            tf.summary.scalar('ad_loss_real', self.ad_loss_real)
            tf.summary.scalar('ad_loss_fake',self.ad_loss_fake)
            tf.summary.scalar('constraint_loss', self.constraint_loss)
            tf.summary.scalar('ad_loss_syn', self.ad_loss_syn)

        if True:
            tf.summary.image('image0/input1', tf.expand_dims(util.restore_img(self.batch_data[0][:,:,::-1]), 0))
            tf.summary.image('image0/input2',
                             tf.expand_dims(util.restore_img(self.batch_data[int(self.batch_size / 2)][:, :, ::-1]), 0))
            tf.summary.image('image0/decoder1', tf.expand_dims(util.restore_img(self.output_syn[0][:, :, ::-1]), 0))
            tf.summary.image('image0/decoder2',
                             tf.expand_dims(util.restore_img(self.output_syn[int(self.batch_size / 2)][:, :, ::-1]), 0))

        
            tf.summary.image('image1/input1', tf.expand_dims(util.restore_img(self.batch_data[1][:,:,::-1]), 0))
            tf.summary.image('image1/input2',
                             tf.expand_dims(util.restore_img(self.batch_data[int(self.batch_size / 2 + 1)][:, :, ::-1]),
                                            0))
            tf.summary.image('image1/decoder1', tf.expand_dims(util.restore_img(self.output_syn[1][:, :, ::-1]), 0))
            tf.summary.image('image1/decoder2',
                             tf.expand_dims(util.restore_img(self.output_syn[int(self.batch_size / 2 + 1)][:, :, ::-1]),
                                            0))

            
            tf.summary.image('image2/input1', tf.expand_dims(util.restore_img(self.batch_data[2][:,:,::-1]),0))
            tf.summary.image('image2/input2',
                             tf.expand_dims(util.restore_img(self.batch_data[int(self.batch_size / 2 + 2)][:, :, ::-1]),
                                            0))
            tf.summary.image('image2/decoder1', tf.expand_dims(util.restore_img(self.output_syn[2][:, :, ::-1]), 0))
            tf.summary.image('image2/decoder2',
                             tf.expand_dims(util.restore_img(self.output_syn[int(self.batch_size / 2 + 2)][:, :, ::-1]),
                                            0))

    def _loss_compute(self):
        '''
        loss 加权
        :return:
        '''
        # 引入正则项
        self.d_loss = tf.divide(self.ad_loss_real + self.ad_loss_fake, tf.constant(2, dtype=tf.float32))
        self.g_loss = self.constraint_loss * 0.997 + self.ad_loss_syn * 0.003
        tf.summary.scalar('losstotal/total_loss_d', self.d_loss)
        tf.summary.scalar('losstotal/total_loss_g', self.g_loss)







