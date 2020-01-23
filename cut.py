
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

def cutNumber():
    
    # 读取输入图片
    image = cv2.imread("./imgs/source_4.jpg")

    # 将输入图片裁剪到固定大小
    image = imutils.resize(image, height=500)
    # 将输入转换为灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行高斯模糊操作
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 执行边缘检测
    edged = cv2.Canny(blurred, 50, 200, 255)
    # cv2.imwrite('edge.png', edged)
    cv2.imshow('edge.png', edged)

    # 在边缘检测map中发现轮廓
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # 根据大小对这些轮廓进行排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    # 循环遍历所有的轮廓
    for c in cnts:
	    # 对轮廓进行近似
	    peri = cv2.arcLength(c, True)
	    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	    # 如果当前的轮廓有4个顶点，我们返回这个结果，即LCD所在的位置
	    if len(approx) == 4:
		    displayCnt = approx
		    break

    # 应用视角变换到LCD屏幕上
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    cv2.imshow('warped.png', warped)
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    cv2.waitKey(0)

cutNumber()