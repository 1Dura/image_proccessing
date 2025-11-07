import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import math

# 1. Для изображения sar_3.jpg найти наиболее протяженный участок
# (выделить линии при помощи преобразования Хафа)


image = cv2.imread('../image_proccessing/lab3/sar_3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 



bin_img = copy.deepcopy(image_gray)
border  = 80
bin_img[image_gray < border] = 0
bin_img[image_gray >= border] = 255

canny = cv2.Canny(bin_img,0,250,apertureSize = 3)
lines = cv2.HoughLines(canny, 1, np.pi / 180, 110)

line_image=image.copy()
if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(line_image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

plt.imshow(line_image)
plt.title('Увэренная в себе линия')
plt.show()

# 2. Для изображения sar_3.jpg провести исследование алгоритмов бинаризации, выделить участок дорожной полосы.

def image_diff(image1, title1, image2, title2):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1)

    plt.subplot(1,2,2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)

    plt.show()

# Точечная бинаризация

bin_img = copy.deepcopy(image_gray)
T  = 80
bin_img[image_gray < T] = 0
bin_img[image_gray >= T] = 255
image_diff(image, 'Исходник', bin_img, 'Точечная бинаризация')

# Бинаризация Отсу

_,th2 = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

image_diff(image, 'Исходник', th2, 'Бинаризация Отсу')

# Адаптивная бинаризация

th3 = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,71,21)

image_diff(image, 'Исходник', th3, 'Адаптивная бинаризация')


# оператор Собеля

scale = 1
delta = 0
ddepth = cv2.CV_16S
grad_x = cv2.Sobel(image_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(image_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
th3=(grad_x - grad_x.min())*255
image_diff(image, 'Исходник', th3, 'Оператор Собеля')

# Canny

edges = cv2.Canny(image_gray,100,200)
image_diff(image, 'Исходник', edges, 'Canny')