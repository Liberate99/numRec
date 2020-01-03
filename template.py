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
    temList = pointList
    pointList.sort()
    temList.sort()
    for point_1 in pointList:
        for point_2 in pointList:
            dis = distanceOF2points(point_1, point_2)
            if dis < w:
                temList.remove(point_2)
    resultList = temList
    return resultList

def template(templateImage, sourceImage):

    # 模板图片 - 灰度图
    tpl_0 = cv.imread('./imgs/0.jpg', 0)
    tpl_0_show = cv.resize(tpl_0, (0, 0), fx=1.0, fy=1.0, interpolation=cv.INTER_NEAREST)
    cv.imshow('0', tpl_0_show)
    tpl_f = cv.imread('./imgs/F.jpg', 0)
    # 目标图片
    target = cv.imread('./imgs/source_2.jpg')
    # 显示目标图片
    target_show = cv.resize(target, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    cv.imshow('target', target_show)
    # 目标图片转换为灰度图
    source_img = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

    h, w = tpl_0.shape[:2]

    # 2.归一化相关模板匹配
    # TM_SQDIFF_NORMED: 归一化平方差匹配(平方差匹配CV_TM_SQDIFF：用两者的平方差来匹配，最好的匹配值为0)
    # TM_CCORR_NORMED:  归一化相关匹配(相关匹配CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好)
    # TM_CCOEFF_NORMED: 归一化相关系数匹配(相关系数匹配CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配)

    threshold = 0.6

    # 0
    res_0 = cv.matchTemplate(source_img, tpl_0, cv.TM_CCOEFF_NORMED)
    loc = np.where(res_0 >= threshold)  # 匹配程度大于%80的坐标y,x
    results = zip(*loc[::-1])
    pointList = processPointList(list(results), w)
    pointList = processPointList(pointList, w)

    print(len(pointList))
    for point in pointList:
        print(point)

    for pt in pointList:  # *号表示可选参数
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

def intro():
    # 1.读入原图和模板
    template()

intro()
cv.waitKey(0)
cv.destroyAllWindows()