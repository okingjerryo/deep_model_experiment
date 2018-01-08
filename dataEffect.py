import numpy as np
import descriminator.args as args
import tensorflow as tf
import os.path
import glymur
import cv2
import descriminator.util as util
def convert_2d(r):
    # 添加均值为 0, 标准差为 64 的加性高斯白噪声
    s = r + np.random.normal(0, 64, r.shape)
    if np.min(s) >= 0 and np.max(s) <= 255:
        return s
    # 对比拉伸
    s = s - np.full(s.shape, np.min(s))
    s = s * 255 / np.max(s)
    s = s.astype(np.uint8)
    return s


def gass_process(image):
    s_dsplit = []
    for d in range(image.shape[2]):
        rr = image[:, :, d]
        ss = convert_2d(rr)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s

def process_pic_one(path):
    image = cv2.imread(path)
    this_file = os.path.basename(path)
    pure_name = this_file.split(".")[0]
    cv2.imwrite(args.oraginal_dir+"/"+pure_name+".jpg",image)
    # 高斯白性噪声
    im_g = gass_process(image)
    save_dir = args.goss_dir+"/"
    util.ensure_dir(save_dir)
    cv2.imwrite(args.goss_dir+"/"+pure_name+".jpg",im_g)
    # 高斯模糊
    im_m = cv2.GaussianBlur(image, args.goss_m_kernel_size, args.goss_m_sigma)
    save_dir = args.m_dir + "/"
    util.ensure_dir(save_dir)
    cv2.imwrite(args.m_dir+"/"+pure_name+".jpg",im_m)
    # jpeg 压缩
    save_dir = args.jpeg_dir + "/"
    util.ensure_dir(save_dir)
    cv2.imwrite(args.jpeg_dir+"/"+pure_name+".jpg",
                image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_com_radio])
    # jpeg 2000压缩 由于是 jp2格式 暂时忽略
    # 注意必须要转换色彩空间
    # image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #
    # save_dir = args.jpeg_2000_dir + "/"
    # util.ensure_dir(save_dir)
    # glymur.Jp2k(args.jpeg_2000_dir+"/"+pure_name+".jp2",data=image_bgr)
