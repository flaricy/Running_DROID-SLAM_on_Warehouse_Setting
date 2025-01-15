'''
    This script is used to extract images from .bag files in an Ubuntu environment
    without using ROS. To use this script please change the path to your own ones.

    The output images are saved in `./images/` here.

'''


import rosbag
import os
import numpy as np
import cv2


if __name__ == '__main__':
    bag_name = 'complex_sub1.bag' #修改为实际包名称
    bag_dir = '/root/autodl-tmp/' #修改为实际路径文件夹
    img_dir = os.path.join(bag_dir, 'images') #修改为需要的输出位置
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    bag_file = os.path.join(bag_dir, bag_name)
    bag = rosbag.Bag(bag_file)
    index = 0
    img_name = os.path.join(img_dir, '{:0>5d}.jpg')

    # msg_types, topics = bag.get_type_and_topic_info()
    # print(topics)

    for topic, msg, t in bag.read_messages(topics='/camara/color/image_raw'):
        header = msg.header
        header_seq = header.seq
        stamp_sec = header.stamp.secs
        stamp_nsec = header.stamp.nsecs
        data = msg.data #bytes
        img = np.frombuffer(data, dtype=np.uint8) #转化为numpy数组
        img = img.reshape(msg.height, msg.width, 3) #3通道图片
        cv2.imwrite(img_name.format(index), img) #保存
        # print('{:0>5d} {} {} {}'.format(index, header_seq, stamp_sec, stamp_nsec))
        index += 1
