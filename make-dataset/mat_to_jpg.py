import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as scio
from PIL import Image

# 帮助训练UNET模型，训练数据只有一个mask
def get_all_files():
    path = os.path.abspath('.')
    # print(path)
    path1 = path + '/groundTruth'
    return os.listdir(path1)  # 打印文件夹内的文件名称


def mat_to_jpg(files_list):
    for i in range(len(files_list)):
        path = r'./groundTruth/{}'.format(files_list[i])
        mat_data = scio.loadmat(path)
        # print(mat_data)
        array_mat = mat_data['groundTruth'][0][0]['Segmentation']
        array_mat = np.asarray(array_mat)
        array_mat = array_to_binary(array_mat)
        image = Image.fromarray(array_mat).convert("L")  # L为模式
        split_list = files_list[i].split('.')
        image.save("./groundTruth_jpg/{}_mask.jpg".format(split_list[0]))  # 输出图片格式可以自己选择


def array_to_binary(array):
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] == 1:
                array[i][j] = 0
            else:
                array[i][j] = 255
    return array

if __name__ == '__main__':
    files_list = get_all_files()
    mat_to_jpg(files_list)
