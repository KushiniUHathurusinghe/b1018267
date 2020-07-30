import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("Images/LetterD/D1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Images/LetterD/D2.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("Images/LetterD/D3.jpg", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("Images/LetterD/D4.png", cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread("Images/LetterD/D5.jpg", cv2.IMREAD_GRAYSCALE)
img6= cv2.imread("Images/LetterD/D6.jpg", cv2.IMREAD_GRAYSCALE)

img11 = cv2.imread("Images/LetterD/D1.png")
img22 = cv2.imread("Images/LetterD/D2.png")
img33 = cv2.imread("Images/LetterD/D3.jpg")
img44 = cv2.imread("Images/LetterD/D4.png")
img55 = cv2.imread("Images/LetterD/D5.jpg")
img66= cv2.imread("Images/LetterD/D6.jpg")

# f = np.fft.fft2(img)
# fShift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fShift))
# magnitude_spectrum = np.asarray(magnitude_spectrum,dtype=np.uint8)

# cv2.imshow("magnitude spectrum",magnitude_spectrum)
# cv2.imshow("Image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.subplot(121),plt.imshow(img66, cmap = 'gray')
plt.title('Input Image 6'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img6, cmap = 'gray')
plt.title('Gray Scaled Input Image 6'), plt.xticks([]), plt.yticks([])
# plt.subplot(121),plt.imshow(img3, cmap = 'gray')
# plt.title('Input Image 3'), plt.xticks([]), plt.yticks([])
# plt.subplot(124),plt.imshow(img4, cmap = 'gray')
# plt.title('Input Image 4'), plt.xticks([]), plt.yticks([])
# plt.subplot(121),plt.imshow(img5, cmap = 'gray')
# plt.title('Input Image 5'), plt.xticks([]), plt.yticks([])
# plt.subplot(121),plt.imshow(img6, cmap = 'gray')
# plt.title('Input Image 6'), plt.xticks([]), plt.yticks([])
plt.show()