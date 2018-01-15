import os

# sys using gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_image = "resource/00283.bmp"
oraginal_dir = "/home/huangfei/db/huatielu/test/original"
goss_dir = "/home/huangfei/db/huatielu/test/goss"
m_dir = "/home/huangfei/db/huatielu/test/m_goss"
jpeg_dir = "/home/huangfei/db/huatielu/test/jpeg"
jpeg_2000_dir = "resource/jpeg2000"
train_dir = "/home/huangfei/db/exploration_database_and_code/test"
#effect.py
goss_m_kernel_size = (5, 5)
goss_m_sigma = 1.5
jpeg_com_radio = 8