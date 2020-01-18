import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import cv2 as cv
    import numpy as np

    # Q1A

    img = cv.imread('img.jpg', 0)
    img = img[:1200, :1200]  # crop to 700 x 700
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[int(crow - 50):int(crow + 50), int(ccol - 50):int(ccol + 50)] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    magnitude_spectrum_mask = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks(), plt.yticks()
    plt.subplot(222), plt.imshow(magnitude_spectrum_mask, cmap='gray')
    plt.title('Box Filter'), plt.xticks(), plt.yticks()
    plt.subplot(223), plt.imshow(img_back, cmap='gray')
    plt.title('Lowpass Filter Output'), plt.xticks(), plt.yticks()

    # Q1B

    f_ishift_slice = np.fft.ifftshift(fshift[550:650, 550:650, :])
    img_back = cv2.idft(f_ishift_slice)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    plt.subplot(224), plt.imshow(img_back, cmap='gray')
    plt.title('Sliced Lowpass Filter Output'), plt.xticks(), plt.yticks()
    plt.show()
