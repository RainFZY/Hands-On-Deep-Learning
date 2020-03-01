# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np


def add_region_channel(im_orig, scale, sigma, min_size): # img, scale=500, sigma=0.9, min_size=10
    """
        应用Felzenswalb算法，根据传入的参数，在第三维上加一层/加一个通道，放不同的region
        rerurn: 512*512*4的图
    """

    # 产生一层区域的mask
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)

    # 把mask合并到原图第三维的第四个通道
    im_orig = np.append(im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def cal_color_sim(r1, r2):
    """
        计算两个区域颜色的相似度
        return: double
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def cal_texture_sim(r1, r2):
    """
        计算两个区域纹理的相似度
        return: double
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def cal_size_sim(r1, r2, img_size):
    """
        计算两个区域尺寸的相似度
        return：double
    """
    return 1.0 - (r1["size"] + r2["size"]) / img_size


def cal_fill_sim(r1, r2, img_size):
    """
        计算两个区域交叠的相似度
        return：double
    """
    # 能包含两个区域的最小矩形区域
    BBsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (BBsize - r1["size"] - r2["size"]) / img_size


def sum__sim(r1, r2, img_size):
    """
        计算两个区域的相似度
        return：double
    """
    return (cal_color_sim(r1, r2) + cal_texture_sim(r1, r2)
            + cal_size_sim(r1, r2, img_size) + cal_fill_sim(r1, r2, img_size))


def get_color_hist(img):
    """
        计算输入区域的颜色直方图
        return size: BINS * COLOUR_CHANNELS(3)
    """
    BINS = 25
    hist = np.array([])

    for colour_channel in (0, 1, 2):

        # 依次提取每个颜色通道
        c = img[:, colour_channel]

        # 计算每个颜色的直方图，加入到结果中
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])

    # 标准化
    hist = hist / len(img)

    return hist


def LBP_texture(img):
    """
        用LBP(局部二值模式)计算整幅图的纹理梯度,提取纹理特征
        return: 512*512*4
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    # 512*512*4
    return ret


def get_texture_hist(img):
    """
        计算每个区域的纹理直方图
        输出直方图的大小：BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10

    hist = np.array([])

    for colour_channel in (0, 1, 2):

        # mask by the colour channel
        fd = img[:, colour_channel]

        # 计算每个方向的直方图，加入到结果中
        hist = np.concatenate([hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])

    # 标准化
    hist = hist / len(img)

    return hist


def get_regions(img):
    """
        从图像中提取区域，包括区域的尺寸，颜色和纹理特征
        return: 包含min_x,min_y,max_x,max_y,labels,size,hist_c,hist_t这些key的区域字典
    """
    R = {} # 候选区域列表，R的key是区域四个点的

    hsv = skimage.color.rgb2hsv(img[:, :, :3]) # rgb转hsv

    # 计算区域位置、角点坐标
    for y, i in enumerate(img): # 遍历,img是(x,y,(r,g,b,l))
        for x, (r, g, b, l) in enumerate(i): # 遍历l，从0到285
            # 将所有分割区域的外框加到候选区域列表中
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff, # 把min先设成最大，max先设成最小
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # 更新边界
            if R[l]["min_x"] > x: # 新的x比原来x的最小值更小
                R[l]["min_x"] = x # x的最小值更新为新的x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # 计算图像纹理梯度
    tex_grad = LBP_texture(img)

    # 计算每个区域的颜色直方图
    for k, v in list(R.items()): # R中的每一组key, value

        masked_pixels = hsv[:, :, :][img[:, :, 3] == k] # 找出某一key对应区域所有点的h,s,v值

        R[k]["size"] = len(masked_pixels / 4) # 某一key对应区域所有点的个数
        R[k]["hist_c"] = get_color_hist(masked_pixels) # 颜色直方图
        R[k]["hist_t"] = get_texture_hist(tex_grad[:, :][img[:, :, 3] == k]) # 纹理直方图

    # 返回的R依然有0-285这286个key，但是每个key下的字典中除了min_x,min_y,max_x,max_y,labels这几个key,
    # 新增了size,hist_c,hist_t这些key
    return R


def get_region_neighbors(regions):
    """
        通过计算每个区域与其余的所有区域是否有相交，来判断是不是邻居
        input: 区域字典R
        output: neighbor列表，列表中的每个元素以(a,b)形式，a和b分别是两个有重叠区域的key的字典
    """

    # 检测a,b长方形区域是否存在交叉重叠部分
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R_list = list(regions.items()) # 把传进来的R以列表形式表示

    neighbours = []
    for cur, a in enumerate(R_list[:-1]):
        for b in R_list[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours



def merge_regions(r1, r2):
    """
        input: 区域字典R中的两个key，也就是两个区域
        output: 合并后的新的区域，代表一个新的key
    """
    new_size = r1["size"] + r2["size"]
    # 合并后的新的区域字典
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt



def selective_search(
        im_orig, scale=1.0, sigma=0.8, min_size=50): # img, scale=500, sigma=0.9, min_size=10
    """
    input:
        im_orig: Input image
        scale: int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma: float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size: int
            Minimum component size for felzenszwalb segmentation.
    output:
        img : image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    """

    # 增加第四层区域标签，[r,g,b,(region)]，标签值为0-285
    img = add_region_channel(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    img_size = img.shape[0] * img.shape[1] # 512 * 512

    # 返回的R依然有0-285这286个key，但是每个key下的字典中除了min_x,min_y,max_x,max_y,labels这几个key,
    # 新增了size,hist_c,hist_t这些key
    R = get_regions(img)

    # extract neighbouring information
    # 返回一个列表，列表中的每个元素以(a,b)形式，a和b分别是两个有重叠区域的key的字典
    neighbours = get_region_neighbors(R)

    # calculate initial similarities
    S = {} # 相似度集
    # ai,bi是region label(0-285)，ar,br是其对应的矩阵
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = sum__sim(ar, br, img_size) # 计算相似度，(ai,bi)对应一个相似度值

    """
        Hierarchical Grouping Algorithm 
        层次分组算法
    """
    while S != {}:
        # 获得相似度最高的两个区域标签
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # 开辟一个新的key，存放合并两个最相似区域后的区域
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # 把合并的两个区域的标签，加入待删除列表
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # 从S里面移除所有关于合并的两个区域的相似度
        for k in key_to_delete:
            del S[k]

        # 计算新形成的区域的相似度，更新相似度集
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = sum__sim(R[t], R[n], img_size)

    # 从所有的区域R中抽取目标定位框框L，放到新的列表中，返回
    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions


img = skimage.data.astronaut()
selective_search(img, scale=500, sigma=0.9, min_size=10)