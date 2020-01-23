import cv2 as cv
import numpy as np
import math


# gamma变换
def adjust_gamma(src, gamma=0.25):
    scale = float(np.iinfo(src.dtype).max - np.iinfo(src.dtype).min)
    dst = ((src.astype(np.float32) / scale) ** gamma) * scale
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    return dst

# 获取二值化图片
def getBinaryImg(source):
    _threshold = 0
    # 取阈值
    for index in range(len(source)):
        if _threshold < max(source[index]):
            _threshold = max(source[index])
    _threshold = _threshold - 10
    # print("阈值： " + str(_threshold))
    retval, result = cv.threshold(source, _threshold, 255, cv.THRESH_BINARY)
    return result

# 处理图片
def processIMG(source):

    # 先gamma变换
    #img_gammar = adjust_gamma(source)
    #cv.imshow('img_gammar', img_gammar)

    # 取g通道 后获取二值化图像
    r, g, b = cv.split(source)

    #result_r = getBinaryImg(r)
    #cv.imshow('binary_result_r', cv.resize(result_r, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))
    result_g = getBinaryImg(g)
    cv.imshow('1binary_g', cv.resize(result_g, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))
    #result_b = getBinaryImg(b)
    #cv.imshow('result_b', cv.resize(result_b, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))

    # 开运算去噪音
    # 先腐蚀后膨胀叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    result_morphology = cv.morphologyEx(result_g, cv.MORPH_OPEN, kernel)
    cv.imshow('2result_morphology', cv.resize(result_morphology, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))

    # 消除大光斑
    # 通过多次腐蚀操作找出大光斑区域， 再将大光斑区域进行多次膨胀操作
    result_erode_fake_area = cv.erode(result_morphology, kernel, iterations=7)
    cv.imshow('result_erode_fake_area', cv.resize(result_erode_fake_area, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))
    # 对fake_area进行膨胀操作
    result_dilate_erode_fake_area = cv.dilate(result_erode_fake_area, kernel, iterations=7)
    cv.imshow('result_dilate_erode_fake_area', cv.resize(result_dilate_erode_fake_area, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))
    # 原图腐蚀后减去大光斑区域
    result_clean = cv.erode(result_morphology, kernel, iterations=1) - result_dilate_erode_fake_area
    cv.imshow('3result_clean', cv.resize(result_clean, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))
    # 开运算去噪音
    result_morphology_final = cv.morphologyEx(result_clean, cv.MORPH_OPEN, kernel)
    cv.imshow('4result_morphology_final', cv.resize(result_morphology_final, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))

    # # 膨胀+腐蚀
    # result_dilate = cv.dilate(result_morphology_final, kernel, iterations=6)
    # result = cv.erode(result_dilate, kernel, iterations=4)
    # cv.imshow('result', cv.resize(result, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))

    cv.waitKey(0)
    return result_morphology_final


# 主函数
if __name__ == '__main__':
    print("pretreat")
    # 目标图片
    target = cv.imread('./imgs/source_4.jpg')
    
    # 处理图片
    binaryImg = processIMG(target)

    # 显示图片
    cv.imshow('target', cv.resize(target, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST))
    cv.imshow('binaryImg', binaryImg)
    cv.waitKey(0)


