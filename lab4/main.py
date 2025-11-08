import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


image = cv2.imread('../image_proccessing/lab4/sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

def image_diff(image1, title1, image2, title2):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1)

    plt.subplot(1,2,2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)

    plt.show()

def homo_average(img, mask, point, T):
    av_val = img[mask > 0].sum() / np.count_nonzero(img[mask > 0])
                                                            
    if abs(av_val - img[point]) <= T:
        return True
    
    return False

def region_growing(image, seed_point,homo_fun,r, T):
    mask = np.zeros(image_gray.shape, np.uint8)
    mask[seed_point] = 1
    count = 1
    while count > 0:
        count = 0
        local_mask = np.zeros(image_gray.shape, np.uint8)
        for i in range(r,image.shape[0] - r):
            for j in range(r,image.shape[1] - r):
                if mask[i,j]==0 and mask[i - r:i + r, j-r: j+r].sum() > 0:
                    if homo_fun(image, mask, (i,j), T):
                        local_mask[i,j] = 1
        count = np.count_nonzero(local_mask)
        print(count)
        mask += local_mask
        
    return mask*255

seed_point = (250,150)
mask_avg = region_growing(image_gray,seed_point,homo_average,2, 18)

image_diff(image, 'Исходник', mask_avg, 'Газоны (avg)')



# другой критерий однородности. (медианный)
def homo_median(image, mask, point, T):
    if np.count_nonzero(mask > 0) == 0:
        return False
    
    median_val = np.median(image[mask > 0])
    
    if abs(median_val - image[point]) <= T:
        return True
    
    return False

mask_median = region_growing(image_gray,seed_point,homo_median,2, 18)

image_diff(image, 'Исходник', mask_median, 'Газоны (median)')


# сравнение двух критериев однородности

from skimage.metrics import structural_similarity, mean_squared_error

(ssim_average, diff_average) = structural_similarity(image_gray, mask_avg, full=True)
(ssim_median, diff_median) = structural_similarity(image_gray, mask_median, full=True)

mse_average = mean_squared_error(image_gray, mask_avg)
mse_median = mean_squared_error(image_gray, mask_median)

diff_average = (diff_average * 255).astype("uint8")
diff_median = (diff_median * 255).astype("uint8")

image_diff(mask_avg, '(avg)', mask_median, '(median)')
print(f"average + img: SSIM = {ssim_average:.4f}; MSE = {mse_average:.4f}")
print(f"median + img: SSIM = {ssim_median:.4f}; MSE = {mse_median:.4f}")

# пальмы

image = cv2.imread('../image_proccessing/lab4/palm_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

ret, thresh = cv2.threshold(image_gray,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(thresh, cmap="gray")

dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5) 
plt.imshow(dist, cmap="gray")

ret, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, cv2.THRESH_BINARY) 
plt.imshow(sure_fg, cmap="gray")

sure_fg = sure_fg.astype(np.uint8)
ret, markers = cv2.connectedComponents(sure_fg) 
plt.imshow(markers, cmap="gray")

markers = cv2.watershed(image, markers)
image_diff(image, 'Исходник', markers, len(np.unique(markers)))
