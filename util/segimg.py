import os
import numpy as np
import nibabel as nib
import imageio
import matplotlib
from nibabel.viewers import OrthoSlicer3D
from matplotlib import pylab as plt


def read_niifile(niifilepath):  # 读取niifile文件
    img = nib.load(niifilepath)  # 下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()  # 获取niifile数据
    return img_fdata


def save_fig(niifilepath, savepath):  # 保存为图片
    fdata = read_niifile(niifilepath)  # 调用上面的函数，获得数据
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量，第四维）
    for k in range(z):
        silce = fdata[:, :, k]  # 三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepath, '{}.png'.format(k)), silce)
        # 将切片信息保存为png格式


if __name__ == '__main__':
    # niifilepath = 'PP01_008-n3-fli-resize.nii.gz'
    niifilepath = 'PP01_008-TissueA-final.nii.gz'
    # savepath = 'train'
    savepath = 'label'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    save_fig(niifilepath, savepath)