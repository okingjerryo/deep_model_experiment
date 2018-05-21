import tensorflow as tf
from os import path
import op
import args
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import dlib
import os
# 用于识别不同的操作图片集
TRAIN_DATASET_TYPE = 1
VAILD_DATASET_TYPE = 0

DATA_HANDEL = tf.placeholder(dtype=tf.string, shape=[], name='dataset_handle')

# 需要自己去网上下载 shape_predictor_68_face_landmarks.dat 文件
predictor_path = "resource/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


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
    if dataset_type == TRAIN_DATASET_TYPE:
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


def _read_trainImg_to_string(GT_filepath,dataset_type):
    # 因为原图是jpg格式,为了识别使用CV2先转成了image矩阵后再编码为 tf string
    img_mat = cv2.imread(GT_filepath)
    input_img = tf.image.encode_jpeg(img_mat, quality=100)

    if dataset_type == TRAIN_DATASET_TYPE:
        noise_name = path.basename(GT_filepath)
        noise_dir = path.dirname(args.IMG_HAZE_NOISE_PATH)
        noise_path = noise_dir+'/'+noise_name
        noise_mat = cv2.imread(noise_path)
        noise_img = tf.image.encode_jpeg(noise_mat, quality=100)

        return {
            'noise_img':noise_img ,
            'GT_img':input_img
        }
    elif dataset_type == VAILD_DATASET_TYPE:
        return {
            'vaild_img':input_img
        }

def _bytes_feature(bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes]))

def write_img_record(filelist,dataset_type):
    print('train_file convert start')
    print('detect',len(filelist),'files')
    record_path = None
    if dataset_type == TRAIN_DATASET_TYPE:
        record_path = args.PATH2RECORD + args.RECORD_TRAIN_NAME
    elif dataset_type == VAILD_DATASET_TYPE:
        record_path = args.PATH2RECORD + args.RECORD_VAILD_NAME
    writer = tf.python_io.TFRecordWriter(record_path)
    with tf.Session(config=args.gpu_option()) as sess:
        for thisfile in tqdm(filelist, ascii=True, desc='write img to tfrecord'):

            if dataset_type==TRAIN_DATASET_TYPE:
                feature = sess.run(_read_trainImg_to_string(thisfile,dataset_type=TRAIN_DATASET_TYPE))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'noise_image': _bytes_feature(feature['noise_img']),
                    'label': _bytes_feature(feature['GT_img'])
                }))
                writer.write(example.SerializeToString())

            elif dataset_type==VAILD_DATASET_TYPE:
                feature = sess.run(_read_trainImg_to_string(thisfile, dataset_type=VAILD_DATASET_TYPE))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'noise_image': _bytes_feature(feature['vaild_img'])
                }))
                writer.write(example.SerializeToString())

        writer.close()
    print('save complete,path:',record_path)


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


# 手动设置裁剪框的大小， 分别表示left, top, right, bottom边框扩大率
rescaleBB = [2.1, 3, 2.1, 2.5]


def save_crop_images(file_list, save_root_path):
    for image_path in file_list:
        print('> crop image', image_path)
        try:
            img = cv2.imread(image_path)

            dets = detector(img, 1)
            if len(dets) == 0:
                print('> Could not detect the face, skipping the image ...', image_path)
                continue
            if len(dets) > 1:
                print('> Process only the first detected face !')
            detected_face = dets[0]
            imcrop = cropByFaceDet(img, detected_face)

            parent_dir, img_name = get_dir_name(image_path)

            cv2.imwrite(image_path, imcrop)
        except KeyboardInterrupt:
            break
        except:
            continue


def get_dir_name(img_path):
    tmp = img_path.split('/')
    return tmp[-2], tmp[-1]


def cropImg(img, tlx, tly, brx, bry, rescale):
    l = float(tlx)
    t = float(tly)
    ww = float(brx - l)
    hh = float(bry - t)

    # Approximate LM tight BB
    h = img.shape[0]
    w = img.shape[1]
    # cv2.rectangle(img, (int(l), int(t)), (int(brx), int(bry)), \
    #     (0, 255, 255), 2)
    # todo 判断 不加黑边
    cx = l + ww / 2
    cy = t + hh / 2
    tsize = max(ww, hh) / 2
    l = cx - tsize
    t = cy - tsize

    # Approximate expanded bounding box
    bl = int(round(cx - rescale[0] * tsize))
    if bl < 0:
        bl = 0
    bt = int(round(cy - rescale[1] * tsize))
    if bt < 0:
        bt = 0

    br = int(round(cx + rescale[2] * tsize))
    if br > w:
        br = w
    bb = int(round(cy + rescale[3] * tsize))
    if bb > h:
        bb = h
    nw = int(br - bl)
    nh = int(bb - bt)
    imcrop = np.zeros((nh, nw, 3), dtype='uint8')

    ll = 0
    if bl < 0:
        ll = -bl
        bl = 0
    rr = nw
    if br > w:
        rr = w + nw - br
        br = w
    tt = 0
    if bt < 0:
        tt = -bt
        bt = 0
    bbb = nh
    if bb > h:
        bbb = h + nh - bb
        bb = h
    imcrop[tt:bbb, ll:rr, :] = img[bt:bb, bl:br, :]
    return imcrop


def cropByFaceDet(img, detected_face):
    return cropImg(img, detected_face.left(), detected_face.top(), \
                   detected_face.right(), detected_face.bottom(), rescaleBB)


if __name__ == '__main__':
    # haze_1_tarin
    # train_file = glob(args.IMG_HAZE_GT_PATH)
    # write_img_record(train_file,dataset_type=TRAIN_DATASET_TYPE)
    # vaild_file = glob(args.IMG_HAZE_VAILD_PATH)
    # write_img_record(vaild_file, dataset_type=VAILD_DATASET_TYPE)
    # haze_1_vaild
    # file_list = glob("/home/uryuo/db/starSketch/img/*/*.*")
    file_list2 = glob("/home/uryuo/db/starSketch/img/*/*/*.*")

    save_crop_images(file_list2, 'crops')
