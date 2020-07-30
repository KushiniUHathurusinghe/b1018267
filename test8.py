import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt



list_images = glob.iglob("Images/LetterA/*")

for image_title in list_images:
    img = cv2.imread(image_title, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fShift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fShift))
    magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
    img_and_magnitude = np.concatenate((img , magnitude_spectrum), axis=1)



   # cv2.imshow(image_title, img_and_magnitude)
    plt.subplot(2,2,2)
    # plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 2, 2)
    # plt.subplot(121)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(image_title), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(image_title+'Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


