import os.path
'''
sunkejia
file loader
'''
import scipy
import os
import tensorflow as tf
from PIL import Image
import random
from skimage import io
from skimage import color
from skimage import transform
import numpy as np
import threading
import queue as Queue
import cv2
import math
from glob import glob
import glob
import sklearn.metrics.pairwise as pw
import copy

#自定义数据读取类
'''
设计思路是假设原图和5类噪声共计6类，原图单独在一个文件夹中，噪声图像在指定的文件夹中
每次读取数据取原图一张再取指定的文件夹中随机选择一种噪声读出图像，制作二元组<Io,In>
'''
class ImageReader_Customize(object):
    def __init__(self,original_path='',noise_path='',data_glob='',input_size=110,output_size=96,output_channal=3,random_crop=True
        ,thread_nums = 4,queue_size=256,batch_size=128,random_images=True):
        #获得文件列表
        self.output_channel = output_channal
        self.task = list()
        self.original_path=original_path
        self.noise_path=noise_path
        self.original_list=glob.glob(os.path.join(original_path,data_glob))
        #获得噪音目录
        self.noise_name=[]
        for _,b,_ in os.walk(noise_path):
            self.noise_name=b
            break
        self.noise_count=len(self.noise_name)
        self.input_size=input_size
        self.output_size=output_size
        self.output_channal=output_channal
        self.batch_size=batch_size
        #随机裁剪还是中心裁剪
        self.random_crop=random_crop
        self.thread_nums=thread_nums

        #图像是否要随机打乱
        self.random_images=random_images
        self.q = Queue.Queue(queue_size)
        self.q_in=Queue.Queue(queue_size)
        #存储原始图像列表用于随机打乱
        self.random_original_list=np.asarray(self.original_list).copy()
        if self.random_images:
            self._random_image()
        self.im_count=len(self.original_list)
        self.epoch_batch=self.im_count*6.0/self.batch_size
        print(self.epoch_batch,'batchs/epoch')

    def _norm_img(self, img):
        '''
        归一化图像0-1
        :param img: 0-255图像
        :return: 0-1图像
        '''
        return np.array(img)/127.5 - 1

    def _im_read(self,path1,path2):
        '''
        读取图像
        :param imagepath:图像路径
        :return: 0-255的图像
        '''
        im1 = cv2.imread(path1)
        im2 = cv2.imread(path2)
        return im1,im2

    def _random_image(self):
        '''
        打乱整个原始数据列表
        :return:
        '''
        rng_state = np.random.get_state()
        np.random.shuffle(self.random_original_list)

    def read_data_batch(self):
        '''
        调取数据batch的方法
        :return:  一个数据batch
        '''
        return self.q.get()


    def _get_images(self,path,path2):
        '''
        读取图像
        :param path:
        :return:
        '''
        im,im2 = self._im_read(path,path2)
        return self._transform(im,im2)

    def _transform(self,image,image2):
        '''
        将图像剪切或放缩
        :param image:
        :return: 处理好的图像
        '''
        if self.random_crop and self.input_size!=self.output_size:
            cropped_image_1,cropped_image_2 = self._random_crop(image,image2)
        elif not self.random_crop and self.input_size!=self.output_size:
            crop_pix = int(round(self.input_size-self.output_size)/2.0)
            cropped_image_1 = scipy.misc.imresize(image[crop_pix:crop_pix+self.output_size,crop_pix:crop_pix+self.output_size]
                                                ,[self.output_size,self.output_size])
            cropped_image_2 = scipy.misc.imresize(
                image2[crop_pix:crop_pix + self.output_size, crop_pix:crop_pix + self.output_size]
                , [self.output_size, self.output_size])
        else:
            cropped_image_1 = cv2.resize(image,dsize=(self.output_size,self.output_size),interpolation=cv2.INTER_CUBIC)
            cropped_image_2 = cv2.resize(image, dsize=(self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
            # todo: norm img 函数未定义
        return  self._norm_img(cropped_image_1.reshape([self.output_size,self.output_size,self.output_channel])), \
                self._norm_img(cropped_image_2.reshape([self.output_size, self.output_size, self.output_channel]))

    def _random_crop(self,images,images2):
        # if images.shape[0]>self.input_size:
        images = cv2.resize(images,dsize=(self.input_size,self.input_size),interpolation=cv2.INTER_CUBIC)
        images2 = cv2.resize(images2, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        # images=images.reshape([self.input_size,self.input_size,self.output_channel])
        # images2 = images2.reshape([self.input_size, self.input_size, self.output_channel])

        offsetmax=self.input_size-self.output_size
        random_w=np.random.randint(0,offsetmax)
        random_h=np.random.randint(0,offsetmax)

        return images[random_w:random_w+self.output_size,random_h:random_h+self.output_size,:], \
               images2[random_w:random_w + self.output_size, random_h:random_h + self.output_size, :]


    #todo: 理解函数
    def _mkdual_batch(self):
        while True:
            imagebatch = np.zeros([self.batch_size, self.output_size, self.output_size, self.output_channel])
            for i in range(int(self.batch_size / 2)):
                im_index = random.randint(0, self.im_count - 1)
                noise_index = random.randint(0, self.noise_count - 1)
                path = self.random_original_list[im_index]
                noise_path = path.replace(self.original_path+"original",
                                          os.path.join(self.noise_path, self.noise_name[noise_index])) # todo:由于目录结构不同 放到其他地方时需要注意
                try:
                    im, im2 = self._get_images(path, noise_path)
                    imagebatch[i, :, :, :] = im
                    imagebatch[i + int(self.batch_size / 2), :, :] = im2

                except:
                    a = 1
                    print('ERRO:_mkdual_bath', path)
            self.q.put((imagebatch))

    def Dual_enqueueStart(self):
        for _ in range(self.thread_nums):
            t_out = threading.Thread(target=self._mkdual_batch,args=())
            t_out.daemon =True
            t_out.start()
            self.task.append(t_out) # 添加了 list

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def restore_img(image):
    return (np.array(image) +1) *127.5