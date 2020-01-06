# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/3 20:24
@Author  : DJ
@File    : main.py
"""
from template import template
import cv2 as cv




def mian():
    print("hello")
    img0 = cv.imread("./imgs/0.jpg")
    img1 = cv.imread("./imgs/F.jpg")
    imgs1 = cv.imread("./imgs/source_1.jpg")
    imgs2 = cv.imread("./imgs/source_2.jpg")
    imgs3 = cv.imread("./imgs/source_3.jpg")

    target_show = cv.resize(target, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    cv.imshow('target', target_show)

    # print("hello")

