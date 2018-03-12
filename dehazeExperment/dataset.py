import tensorflow as tf
from os import path
from dehazeExperment import args
from glob import glob

# 用于识别不同的操作图片集
TRAIN_DATASET_TYPE = 1
VAILD_DATASET_TYPE = 0

def _read_trainImg_to_string(GT_filepath,dataset_type):
    name_tensor = tf.constant(GT_filepath)
    input_img = tf.image.decode_jpeg(name_tensor)
    if dataset_type == TRAIN_DATASET_TYPE:
        noise_name = path.basename(GT_filepath)
        noise_dir = path.dirname(args.IMG_HAZE_NOISE_PATH)
        noise_path = noise_dir+'/'+noise_name
        noise_name_tensor = tf.constant(noise_path)
        noise_img = tf.image.decode_image(noise_name_tensor)

        return {
            'noise_img':noise_img ,
            'GT_img':input_img
        }
    elif dataset_type == VAILD_DATASET_TYPE:
        return {
            'vaild_img':input_img
        }

def _bytes_feature(bytes):
    return tf.train.Feature(bytes_list = tf.train.BytesList([bytes]))

def write_img_record(filelist,dataset_type):
    print('train_file convert start')
    print('detect',len(filelist),'files')

    record_path = args.PATH2RECORD+args.RECORD_TRAIN_NAME
    writer = tf.python_io.TFRecordWriter(record_path)
    with tf.Session() as sess:
        for thisfile in filelist:

            if dataset_type==TRAIN_DATASET_TYPE:
                feature = sess.run(_read_trainImg_to_string(thisfile,dataset_type=TRAIN_DATASET_TYPE))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'noise_image': _bytes_feature(feature['noise_img']),
                    'GT_image':_bytes_feature(feature['GT_img'])
                }))
                writer.write(example.SerializeToString())

            elif dataset_type==VAILD_DATASET_TYPE:
                feature = sess.run(_read_trainImg_to_string(thisfile, dataset_type=VAILD_DATASET_TYPE))
                example = tf.train.Example(features=tf.train.Features(feature={
                'vaild_image': _bytes_feature(feature['noise_img'])
                }))
                writer.write(example.SerializeToString())

        writer.close()
    print('save complete,path:',record_path)

if __name__ == '__main__':
    # haze_1_tarin
    train_file = glob(args.IMG_HAZE_GT_PATH)
    write_img_record(train_file,dataset_type=TRAIN_DATASET_TYPE)
    # haze_1_vaild
