
# 数据集相关
PATH2RECORD = '/home/uryuo/db/ntire/tfRecord/'            # 将数据集转 tfrecord 位置
# 原始图片 glob 路径
IMG_HAZE_GT_PATH = '/home/uryuo/db/ntire/image/haze_1/indoor/trainGT/*.jpg'
IMG_HAZE_NOISE_PATH = '/home/uryuo/db/ntire/image/haze_1/indoor/trainHaze/*.jpg'
IMG_HAZE_VAILD_PATH = '/home/uryuo/db/ntire/image/haze_1/indoor/validateHaze/*.jpg'
# tfrecord 命名
RECORD_TRAIN_NAME = 'haze_1_train.tfrecord'
RECORD_VAILD_NAME = 'haze_1_vaild.tfrecord'