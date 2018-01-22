import os

# sys using gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_image = "resource/00283.bmp"
oraginal_dir = "/home/huangfei/db/huatielu/test/original"
goss_dir = "/home/huangfei/db/huatielu/test/goss_8"
m_dir = "/home/huangfei/db/huatielu/test/m_goss_2x2"
jpeg_dir = "/home/huangfei/db/huatielu/test/jpeg_85"
jpeg_2000_dir = "resource/jpeg2000"
train_dir = "/home/huangfei/db/huatielu/test/original"
#effect.py
goss_m_kernel_size = (3, 3)
goss_m_sigma = 3
jpeg_com_radio = 85
