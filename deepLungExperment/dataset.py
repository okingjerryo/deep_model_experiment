import tensorflow as tf
from os import path
from deepLungExperment import op
from deepLungExperment import args
from glob import glob
from tqdm import tqdm
import cv2
from deepLungExperment.args import a
import numpy as np

# 用于识别不同的操作图片集
TRAIN_DATASET_TYPE = 0
VAILD_DATASET_TYPE = 1
TEST_DATASET_TYPE = 2

DATA_HANDEL = tf.placeholder(dtype=tf.string, shape=[], name='dataset_handle')


def _img_effect_process(img_mat, ram_seed=tf.set_random_seed(op.get_timestamp_now())):
    # todo 1: 确保每次只在noise-GT对中获得图片相同，当前为每个batch相同
    # todo 2: 图片读取时的操作，random crop+resize
    img_crop = tf.random_crop(img_mat, args.IMG_CROP_SIZE, seed=ram_seed)
    return img_crop


def _get_tfrecord_structure(dataset_type):
    '''
        定义tfrecord 以及trainning set features的结构
    :param dataset_type:
    :return: features 给tf.parse_single_example用
    '''
    if dataset_type == TRAIN_DATASET_TYPE or dataset_type == TEST_DATASET_TYPE:
        return {
            'noise_image': tf.FixedLenFeature((), tf.string, default_value=''),
            'label': tf.FixedLenFeature((), tf.string, default_value='')
        }
    elif dataset_type == VAILD_DATASET_TYPE:
        return {
            'noise_image': tf.FixedLenFeature((), tf.string, default_value='')
        }


def _read_img_process(img_example, Example_type):
    features = None
    ## 注意注意！ features中的key value对一定要与tfrecord中的k-v相关对应！
    if Example_type == TRAIN_DATASET_TYPE:
        features = _get_tfrecord_structure(dataset_type=TRAIN_DATASET_TYPE)
        parse_features = tf.parse_single_example(img_example, features)

        noise_img = tf.image.decode_image(parse_features['noise_image'])
        GT_img = tf.image.decode_image(parse_features['label'])
        # 使得两个图片的分割区域一致
        seed = op.get_timestamp_now()
        noise_img = _img_effect_process(noise_img, ram_seed=seed)  # todo：每个图片batch的位置一定
        GT_img = _img_effect_process(GT_img, ram_seed=seed)
        return (noise_img, GT_img)


    elif Example_type == VAILD_DATASET_TYPE:
        features = _get_tfrecord_structure(dataset_type=VAILD_DATASET_TYPE)
        parse_features = tf.parse_single_example(img_example, features)

        vaild_img = tf.image.decode_png(parse_features['noise_image'])
        vaild_img = _img_effect_process(vaild_img)
        return vaild_img


def _read_trainImg_to_string(GT_filepath, dataset_type):
    # 因为原图是jpg格式,为了识别使用CV2先转成了image矩阵后再编码为 tf string

    img_mat = cv2.imread(GT_filepath)
    img_height = np.shape(img_mat)[0]
    img_width = np.shape(img_mat)[1]
    input_img = tf.image.encode_jpeg(img_mat, quality=100)

    if dataset_type == TRAIN_DATASET_TYPE or dataset_type == TEST_DATASET_TYPE:
        noise_name = path.basename(GT_filepath)
        noise_dir = path.dirname(args.IMG_HAZE_NOISE_PATH)
        noise_path = noise_dir + '/' + noise_name
        noise_mat = cv2.imread(noise_path)
        noise_img = tf.image.encode_jpeg(noise_mat, quality=100)

        return {
            'noise_img': noise_img,
            'GT_img': input_img
        }
    elif dataset_type == TEST_DATASET_TYPE:
        return {
            'vaild_img': input_img
        }


def _bytes_feature(bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes]))


def write_img_record(filelist, dataset_type):
    print('file convert start')
    print('detect', len(filelist), 'files')
    record_path = None
    if dataset_type == TRAIN_DATASET_TYPE:
        record_path = args.PATH2RECORD + args.RECORD_TRAIN_NAME
    elif dataset_type == TEST_DATASET_TYPE:
        record_path = args.PATH2RECORD + a.record_test_name
    elif dataset_type == VAILD_DATASET_TYPE:
        record_path = args.PATH2RECORD + a.record_vaild_name
    writer = tf.python_io.TFRecordWriter(record_path)
    with tf.Session(config=args.gpu_option()) as sess:
        # 从文件中分别将几个图片读出来，格式即使Test也存gt（只是实际用不用的问题）
        for thisfile in tqdm(filelist, ascii=True, desc='write img to tfrecord'):
            feature = sess.run(_read_trainImg_to_string(thisfile, dataset_type=dataset_type))
            example = tf.train.Example(features=tf.train.Features(feature={
                'noise_image': _bytes_feature(feature['noise_img']),
                'label': _bytes_feature(feature['GT_img'])
            }))
            writer.write(example.SerializeToString())

        writer.close()
    print('save complete,path:', record_path)


def _init_handle_train():
    train_dataset = tf.data.TFRecordDataset([args.PATH2RECORD + args.RECORD_TRAIN_NAME])

    train_dataset = train_dataset.map(
        lambda x: _read_img_process(x, Example_type=TRAIN_DATASET_TYPE)
    )

    train_dataset = train_dataset.repeat(50)  # 组成一个epoch
    train_dataset = train_dataset.batch(args.TRAIN_BATCH)

    # init handle
    iterator = tf.data.Iterator.from_string_handle(DATA_HANDEL, train_dataset.output_types,
                                                   train_dataset.output_shapes)
    get_next_batch = iterator.get_next()

    train_itor = train_dataset.make_initializable_iterator()
    return {
        'train_itor': train_itor,
        'get_next_batch': get_next_batch
    }


def vaildate_record_useable():
    '''
    获取一个batch数据，并取出头第一对图片。看其是否有效
    :return: null
    '''

    data_op = _init_handle_train()
    get_batch = data_op['get_next_batch']
    train_itor = data_op['train_itor']

    with tf.Session(config=args.gpu_option()) as sess:
        print('vaild train tfrecord:', args.PATH2RECORD + args.RECORD_TRAIN_NAME)
        data_handle = sess.run(train_itor.string_handle())

        for i in range(2):
            train_feed = {DATA_HANDEL: data_handle}
            sess.run(train_itor.initializer, feed_dict=train_feed)
            batch_feature = sess.run(get_batch, feed_dict=train_feed)
            noise_batch, GT_batch = batch_feature
            cv2.imwrite(args.RECORD_ALIABLE_PATH + 'noise' + str(i) + '.jpg', noise_batch[0])
            cv2.imwrite(args.RECORD_ALIABLE_PATH + 'GT' + str(i) + '.jpg', GT_batch[0])
            print('noise batch', i, 'shape', noise_batch[0].shape)
            print('GT batch', i, 'shape', GT_batch[0].shape)


# brige feature to estimator
def get_feature_columns():
    feature_columns = []
    # 选择认证集原因，认证集只包含feature 不包含 label
    for key in _get_tfrecord_structure(VAILD_DATASET_TYPE):
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    return feature_columns


if __name__ == '__main__':
    # chinamm train
    # train_file = glob(args.IMG_HAZE_GT_PATH)
    # write_img_record(train_file,dataset_type=TRAIN_DATASET_TYPE)
    # vaild_file = glob(args.IMG_HAZE_VAILD_PATH)
    # write_img_record(vaild_file, dataset_type=VAILD_DATASET_TYPE)
    # haze_1_vaild
    pass
