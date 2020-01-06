# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/2 12:22
@Author  : DJ
@File    : template.py
"""

import cv2 as cv
import numpy as np
import math

# 计算两点间距离
def distanceOF2points(point1,point2):
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    dis = math.pow(x, 2) + math.pow(y, 2)
    dis = math.pow(dis, 0.5)
    return dis

# 处理点列表
def processPointList(pointList, w):
    print("processPointList")
    print("搜索到一共：" + str(len(pointList)) + " 个点, 控制距离为： " + str(w))
    temList_a = pointList
    temList_a.sort()
    temList_b = pointList
    temList_b.sort()
    result_List = []
    for i, point_a in enumerate(temList_a):
        for j, point_b in enumerate(temList_b):
            if j == 0:
                print("hello")

    result_List = []
    return result_List

# gamma变换
def adjust_gamma(src,gamma=0.25):
    scale = float(np.iinfo(src.dtype).max - np.iinfo(src.dtype).min)
    dst = ((src.astype(np.float32) / scale) ** gamma) * scale
    dst = np.clip(dst,0,255).astype(np.uint8)
    return dst

# 处理图片
def processIMG(source):
    _threshold = 0
    # 先gamma变换
    img_gammar = adjust_gamma(source)
    # 取g通道
    r, g, b = cv.split(img_gammar)
    # 取阈值
    for index in range(len(g)):
        if _threshold < max(g[index]):
            _threshold = max(g[index])
    _threshold = _threshold - 10
    print("阈值： " + str(_threshold))
    retval, g = cv.threshold(g, _threshold, 255, cv.THRESH_BINARY)
    return g

# 匹配模板
def template(templateImage, sourceImage):
    # 模板图片
    processed_tpl = processIMG(templateImage)
    cv.imshow('templateImage', processed_tpl)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    erosion = cv.erode(processed_tpl, kernel)  # 腐蚀
    cv.imshow('erosion', erosion)

    # 目标图片
    processed_target = processIMG(sourceImage)
    # 先腐蚀后膨胀叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域
    opened_target = cv.morphologyEx(processed_target, cv.MORPH_OPEN, kernel)  # 开运算
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # dilated_target = cv.dilate(opened_target, kernel)
    img_rgb_show = cv.resize(opened_target, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    cv.imshow('open', img_rgb_show)
    cv.waitKey(0)

    # 归一化相关模板匹配
    # TM_SQDIFF_NORMED: 归一化平方差匹配(平方差匹配CV_TM_SQDIFF：用两者的平方差来匹配，最好的匹配值为0)
    # TM_CCORR_NORMED:  归一化相关匹配(相关匹配CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好)
    # TM_CCOEFF_NORMED: 归一化相关系数匹配(相关系数匹配CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配)
    threshold = 0.6
    res_0 = cv.matchTemplate(opened_target, erosion, cv.TM_CCOEFF_NORMED)
    loc = np.where(res_0 >= threshold)  # 匹配程度大于%80的坐标y,x
    results = zip(*loc[::-1])

    h, w = processed_tpl.shape[:2]
    # 筛选两遍
    pointList = processPointList(list(results), w)
    # pointList = processPointList(pointList, w)
    print(pointList)

    return pointList

if __name__ == '__main__':

    # 1.读入原图和模板
    # 模板图片 - 灰度图
    tpl_0 = cv.imread('./imgs/0.jpg')
    tpl_f = cv.imread('./imgs/F.jpg')
    # 目标图片
    target = cv.imread('./imgs/source_1.jpg')
    # 显示目标图片
    target_show = cv.resize(target, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
    cv.imshow('target', target_show)

    # 2.分别进行模板匹配
    # 0 需传入灰度图
    point_0_list = template(tpl_0, target)
    if len(point_0_list) > 0:
        print("找到 0 ：" + str(len(point_0_list)) + "个")
        cv.waitKey(0)
        h, w = tpl_0.shape[:2]
        for pt in point_0_list:  # *号表示可选参数
            right_bottom = (pt[0] + w, pt[1] + h)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # x1, y1------
            # |           |
            # |           |
            # |           |
            #  --------x2, y2
            cv.rectangle(target, pt, right_bottom, (0, 245, 255), 2)
            img_rgb_show = cv.resize(target, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
            cv.imshow('match-result', img_rgb_show)
    # f
    point_f_list = template(tpl_f, target)
    if len(point_f_list) > 0:
        print("找到 f：" + str(len(point_f_list)) + "个")
        h, w = tpl_f.shape[:2]
        for pt in point_f_list:  # *号表示可选参数
            right_bottom = (pt[0] + w, pt[1] + h)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # x1, y1------
            # |           |
            # |           |
            # |           |
            #  --------x2, y2
            cv.rectangle(target, pt, right_bottom, (0, 245, 255), 2)
            img_rgb_show = cv.resize(target, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
            cv.imshow('match-result', img_rgb_show)

    # img = cv.imread("./imgs/0.jpg")
    # r, g, b = cv.split(img)
    # cv.imshow("test1", g)
    #
    #
    # print(g)
    # thre = max(g[0]) + 0
    # print("最大值: "+str(thre))
    # retval, g = cv.threshold(g, thre, 255, cv.THRESH_BINARY)
    # cv.imshow("test2", g)
    cv.waitKey(0)