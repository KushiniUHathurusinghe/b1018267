import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from pip._vendor.urllib3.connectionpool import xrange

list_images = glob.iglob("Images/A*")
# X_data = []
# imageA =[]

for image_title in list_images:
        img = cv2.imread(image_title, cv2.IMREAD_GRAYSCALE)
        # imageAndMagnitude = np.concatenate((image_title, img), axis=1)
        imageAndMagnitude = np.concatenate((), axis=1)
        # imageA.append(img)
        # print(imageA)
        # f = np.fft.fft2(img)
        # fShift = np.fft.fftshift(f)
        # magnitude_spectrum = 20*np.log(np.abs(fShift))
        # magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)
        # img_and_magnitude = np.concatenate((img , magnitude_spectrum), axis=1)
        # X_data.append(img_and_magnitude)
        # print(X_data)
        cv2.imshow(image_title)
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.show()