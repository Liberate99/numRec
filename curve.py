# -*- coding: utf-8 -*-
"""
@Time    : 2020/1/5 20:12
@Author  : DJ
@File    : curve.py
"""
import cv2 as cv
import numpy
from scipy import interpolate

#
# def _find_coefficients(self):
#     polynomials = []
#     curves = [[(0, 0), (79, 52), (176, 209), (255, 255)], [(0, 0), (74, 56), (175, 206), (255, 255)],
#               [(0, 0), (74, 54), (179, 206), (255, 255)], [(0, 0), (49, 79), (202, 181), (255, 255)],
#               [(0, 0), (255, 255)]]
#     for curve in curves:
#         xdata = [x[0] for x in curve]
#         ydata = [x[1] for x in curve]
#
#         p = interpolate.lagrange(xdata, ydata)
#         polynomials.append(p)
#
#     return polynomials
#
#
# def apply_filter(self, filter_name, image_array):
#     if image_array.ndim < 3:
#         raise Exception('Photos must be in color, meaning at least 3 channels')
#     else:
#         def interpolate(i_arr, f_arr, p, p_c):
#             p_arr = p_c(f_arr)
#             return p_arr
#
#             # NOTE: Assumes that image_array is a numpy array
#
#         image_filter = self.filters[filter_name]
#         # NOTE: What happens if filter does not exist?
#         width, height, channels = image_array.shape
#         filter_array = numpy.zeros((width, height, 3), dtype=float)
#
#         p_r = image_filter.get_r()
#         p_g = image_filter.get_g()
#         p_b = image_filter.get_b()
#         p_c = image_filter.get_c()
#
#         filter_array[:, :, 0] = p_r(image_array[:, :, 0])
#         filter_array[:, :, 1] = p_g(image_array[:, :, 1])
#         filter_array[:, :, 2] = p_b(image_array[:, :, 2])
#         filter_array = filter_array.clip(0, 255)
#         filter_array = p_c(filter_array)
#
#         filter_array = numpy.ceil(filter_array).clip(0, 255)
#
#         return filter_array.astype(numpy.uint8)

def adjust_gamma(src,gamma=0.5):
    scale = float(numpy.iinfo(src.dtype).max - numpy.iinfo(src.dtype).min)
    dst = ((src.astype(numpy.float32) / scale) ** gamma) * scale
    dst = numpy.clip(dst,0,255).astype(numpy.uint8)
    return dst

if __name__ == '__main__':
    img = cv.imread("./imgs/source_1.jpg")
    img_gammar = adjust_gamma(img)
    img_gammar_show = cv.resize(img_gammar, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
    cv.imshow("gamma变换后(rgb)", img_gammar_show)

    r, g, b = cv.split(img_gammar)
    g_show = cv.resize(g, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
    cv.imshow("gamma变换后(g)", g_show)

    thre = max(g[0]) - 25
    print("最大值: " + str(thre))
    retval, g = cv.threshold(g, thre, 255, cv.THRESH_BINARY)
    img_g_show = cv.resize(g, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
    cv.imshow("阈值二值化", img_g_show)

    cv.waitKey(0)