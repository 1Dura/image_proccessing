import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means
from skimage.metrics import structural_similarity, mean_squared_error

# Зашумить изображение при помощи шума гаусса, постоянного шума.
# Протестировать медианный фильтр, фильтр гаусса, билатериальный фильтр, фильтр нелокальных средних с различными параметрами.
# Выяснить, какой фильтр показал лучший результат фильтрации шума.

def imshow(new_image, new_title):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходник')

    plt.subplot(1,2,2)
    plt.imshow(new_image, cmap='gray')
    plt.title(new_title)

    plt.show()


image = cv2.imread('../image_proccessing/lab2/sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Шум гаусса

mean = 0   # точка отсчета
stddev = 100 # отклонение
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)

imshow(noise_gauss, 'Шум Гаусса')

# Постоянный шум

noise_strength=100
noise = np.random.uniform(-noise_strength, noise_strength, image_gray.shape)
noisy_image = image_gray + noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

imshow(noisy_image, 'Постоянный шум')

def nimshow(new_image, new_title):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Исходник, обработанный постоянным шумом')

    plt.subplot(1,2,2)
    plt.imshow(new_image, cmap='gray')
    plt.title(new_title)

    plt.show()

# Медианный фильтр

image_gauss_median = cv2.medianBlur(noisy_image, 3)
nimshow(image_gauss_median, 'Медианный фильтр')

# Билатериальный фильтр

image_gauss_bilat = cv2.bilateralFilter(noisy_image,9,75,75)
nimshow(image_gauss_bilat, 'Билатериальный фильтр')

# Фильтр нелокальных средних с разными переменными

im1 = cv2.fastNlMeansDenoising(noisy_image, 1000)
nimshow(im1, 'Фильтр нелокальных средних. h = 1000')
im1 = cv2.fastNlMeansDenoising(noisy_image, 250)
nimshow(im1, 'Фильтр нелокальных средних. h = 250')
im1 = cv2.fastNlMeansDenoising(noisy_image, -1000)
nimshow(im1, 'Фильтр нелокальных средних. h = -1000')