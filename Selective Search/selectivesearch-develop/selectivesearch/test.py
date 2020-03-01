# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import time


def main():

    # 导入宇航员图片，原图是512*512*3，第三维是RGB
    img = skimage.data.astronaut()
    img = skimage.data.chelsea()
    img = skimage.data.coffee()
    img = skimage.data.hubble_deep_field()


    '''
        scale：图像分割的集群程度。值越大，意味集群程度越高，分割的越少，获得子区域越大。默认为1
        sigma：图像分割前，会先对原图像进行高斯滤波去噪，sigma即为高斯核的大小。默认为0.8
        min_size：最小的区域像素点个数。当小于此值时，图像分割的计算就停止，默认为20
    '''

    img_lbl, regions = selectivesearch.selective_search(img, scale=250, sigma=0.8, min_size=500)
    # region是一个列表，每一个元素是一个字典，存放每一个区域的信息（rect,size,labels三个key）

    # 创建一个新集合并添加所有区域
    region_rect = set()
    for r in regions:
        region_rect.add(r['rect']) 

    # 在原图上绘制矩形框
    # 生成1行1列，大小为6*6的一个字图，fig用来生成一个新的图，ax用来控制子图
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    i = 1
    for x, y, w, h in region_rect:
        print("Region",i,":",regions[i])
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='yellow', linewidth=2)
        ax.add_patch(rect)
        i+=1

    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    run_time = end_time - start_time
    print("run time =",run_time,"s")