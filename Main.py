import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

imageList = glob.iglob("Images/LetterA/*")
# imageList = glob.iglob("Images/LetterD/*")
print(imageList)

for imageTitle in imageList:
        grayImage = cv2.imread(imageTitle, cv2.IMREAD_GRAYSCALE)
        fftImage = np.fft.fft2(grayImage)
        fShift = np.fft.fftshift(fftImage)
        magnitudeSpectrum = 20 * np.log(np.abs(fShift))
        magnitudeSpectrum = np.asarray(magnitudeSpectrum, dtype=np.uint8)
        imageAndMagnitude = np.concatenate((grayImage, magnitudeSpectrum), axis=1)

        # plt.subplot(121), plt.imshow(grayImage, cmap='gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(magnitudeSpectrum, cmap='gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

        scaledImage = cv2.resize(imageAndMagnitude, (432, 323))
        cv2.imshow(imageTitle, scaledImage)

        # plt.show()

# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

cv2.waitKey(0)
cv2.destroyAllWindows()




