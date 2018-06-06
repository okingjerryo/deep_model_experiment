import tensorflow as tf
import argparse


# limit gpu mem useage
def gpu_option():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.ConfigProto(gpu_options=gpu_options)


# tf args
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", type=str, required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--image_size", default=512, type=int, help="input img size")
parser.add_argument("--img_crop_size", default=256, type=int, help="crop target size")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=32, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
# dataset options
parser.add_argument("--path_to_record", default="/home/uryuo/db/ChinaMM/ppchallenge2018/tfRecord",
                    help="the path navgate to tfrecord dir")
parser.add_argument("--img_haze_gt_path", default="/home/uryuo/db/ChinaMM/ppchallenge2018/dataset/train_img_tpg/GT/*.*",
                    help="path to ground true images")
parser.add_argument("--img_haze_noise_path",
                    default='/home/uryuo/db/ChinaMM/ppchallenge2018/dataset/train_img_tpg/Composed/*/*.*',
                    help="path to noise path")
parser.add_argument("--img_haze_vaild_path",
                    default='/home/uryuo/db/ChinaMM/ppchallenge2018/dataset/vaild_img_tpg/Composed/*/*.*',
                    help="path to vaild path")
parser.add_argument("--img_haze_test_path",
                    default='/home/uryuo/db/ChinaMM/ppchallenge2018/dataset/test_img_tpg/Composed/*/*.*',
                    help="path to vaild path")
parser.add_argument("--record_train_name", default="chinamm_1_train.tfrecord", help="train tfrecord file name")
parser.add_argument("--record_vaild_name", default="chinamm_1_vaild.tfrecord", help="vaild tfrecord file name")
parser.add_argument("--record_test_name", default="chinamm_1_test.tfrecord", help="test tfrecord file name")
parser.add_argument("--train_batch", default=50, help="trainning batch size")
a = parser.parse_args()
# 数据集相关
PATH2RECORD = a.path_to_record  # 将数据集转 tfrecord 位置
# 原始图片 glob 路径
IMG_HAZE_GT_PATH = a.img_haze_gt_path
IMG_HAZE_NOISE_PATH = a.img_haze_noise_path
IMG_HAZE_VAILD_PATH = a.img_haze_vaild_path
# tfrecord 命名
RECORD_TRAIN_NAME = a.record_train_name
RECORD_VAILD_NAME = a.record_vaild_name
# record验证图片存储LUJING
RECORD_ALIABLE_PATH = 'resource/'
# network
EPS = 1e-12
TRAIN_BATCH = a.train_batch
IMG_CROP_SIZE = [a.img_crop_size, a.img_crop_size, 3]
INPUT_SIZE = [a.image_size, a.image_size, 3]
