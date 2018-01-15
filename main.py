'''
sunkejia
main file
python main.py
'''
import datetime
import glob
import sys
import tensorflow as tf

import args as args
import dataEffect as effect
from train import Trainer
from util import ImageReader_Customize

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 256, 'tarin_batch_size for one gpu')
flags.DEFINE_integer("sample_batch_size",20, 'test,sample batch_size for one gpu,一般比pose_c相关')  # 保证可以被2、gpu数整除
flags.DEFINE_string('root_path', './logdir_caspeal', 'root path')
flags.DEFINE_integer("input_size", 32, 'train_size')
flags.DEFINE_integer("input_channel", 3, 'train_channel')
flags.DEFINE_integer("output_size", 32, 'output_size')
flags.DEFINE_integer('src_size', 300, 'crop op in this size')
flags.DEFINE_integer("output_channel", 3, 'out_put_channel')
flags.DEFINE_boolean('train', True, 'if Train or inference')
flags.DEFINE_bool('random_crop', True, 'is random crop??')
flags.DEFINE_float("beta1", 0.5, 'moment--m')
flags.DEFINE_float("d_learning_rate", 0.0002, 'base learning rate')
flags.DEFINE_float("g_learning_rate", 0.0002, 'base_learning rate')
flags.DEFINE_integer('mode', 2, 'GAN mode,2:mean MultiPIE')
flags.DEFINE_float("validation_interval", 10, 'validation interval save the decode images')
flags.DEFINE_integer("epoch", 240, 'train_epoch')

'''image data reader'''
flags.DEFINE_string("datapath", '/home/huangfei/db/huatielu/train/', '')
flags.DEFINE_string("dataglob", '/home/huangfei/db/huatielu/train/original/*.jpg', '')
flags.DEFINE_string("noisepath", '/home/huangfei/db/huatielu/train/noise/', '')
'''test data reader'''
flags.DEFINE_string("test_datapath", '/home/huangfei/db/huatielu/test/', '')
flags.DEFINE_string("test_dataglob", '/home/huangfei/db/huatielu/test/original/*.jpg', '')
flags.DEFINE_string("test_noisepath", '/home/huangfei/db/huatielu/test/noise/', '')


flags.DEFINE_integer('thread_nums', 10, 'data read thread nums')
flags.DEFINE_string('model_name', 'L1GAN', '')
flags.DEFINE_string('newtag', 'zero-2018-1-2-01', 'model version')
# 如果GD不一区训练的话，会出现彩色块
flags.DEFINE_string('discribe', '第一版本','version discribe')
flags.DEFINE_string("gpu_list", '0', "CUDA visiabTruele device")
'''continue train'''
flags.DEFINE_boolean('restored', False, 'finetuning model')
FLAGS = flags.FLAGS
FLAGS(sys.argv)
lab_dir = '{}/logdir_{}'.format(FLAGS.root_path, FLAGS.model_name)
result_save_path = '{}/{}_gpu_{}_v{}'.format(lab_dir, datetime.date.today().strftime("%Y%m%d"), FLAGS.gpu_list,
                                             FLAGS.newtag)#存储路径名称
check_point = '{}'.format('checkpoint')#checkpoint路径
log_path = '{}/{}.log'.format(result_save_path, FLAGS.model_name)#log文件位置
image_save_path = '{}/{}'.format(result_save_path, 'image_synthesis')#训练过程中图片的保存位置
summary_save_path = '{}/summarytotal'.format(lab_dir)#summary存储位置

def main():
    data_reader_train = ImageReader_Customize(original_path=FLAGS.datapath,noise_path=FLAGS.noisepath,data_glob=FLAGS.dataglob,
                                                 input_size = FLAGS.src_size, output_size = FLAGS.input_size,
                                                output_channal=FLAGS.output_channel,batch_size=FLAGS.batch_size,
                                              random_crop=FLAGS.random_crop,thread_nums=FLAGS.thread_nums,
                                              )
    data_reader_test = ImageReader_Customize(original_path=FLAGS.test_datapath,noise_path=FLAGS.test_noisepath,data_glob=FLAGS.test_dataglob,
                                                 input_size = FLAGS.src_size, output_size = FLAGS.input_size,
                                                output_channal=FLAGS.output_channel,batch_size=FLAGS.batch_size,
                                              random_crop=FLAGS.random_crop,thread_nums=FLAGS.thread_nums,
                                             )

    drgan = Trainer(sample_interval=FLAGS.validation_interval,
                    restored=FLAGS.restored, batch_size=FLAGS.batch_size,
                    epoch=FLAGS.epoch, log_dir=lab_dir, d_learning_rate=FLAGS.d_learning_rate,
                    g_learning_rate=FLAGS.g_learning_rate, beta1=FLAGS.beta1,
                    test_batch_size=FLAGS.sample_batch_size, model_name=FLAGS.model_name,
                    output_channel=FLAGS.output_channel,
                    input_size=FLAGS.input_size, input_channel=FLAGS.input_channel, data_loader_train=data_reader_train,
                    gpus_list=FLAGS.gpu_list,
                    check_point=check_point, output_size=FLAGS.output_size,
                    version=FLAGS.newtag,
                    savepath=result_save_path, imagepath=image_save_path, logfile=log_path,
                    summary_dir=summary_save_path,
                    discribe=FLAGS.discribe,
                    data_loader_valid=data_reader_test)
    if FLAGS.train:
        drgan.train()



def pic_process():


    list = glob.glob(args.train_dir+"/*")
    i = 1
    for path in list:
        effect.process_pic_one(path)
        i+=1
        if i % 100 ==0:
            print("process:{}pic".format(i))
        

if __name__ == '__main__':
    print(tf.__path__)
    # pic_process()
    # 如果在分支中看到证明 分支正常
    main()
