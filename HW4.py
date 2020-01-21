import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    import cv2 as cv
    import numpy as np

    # Q1A

    img = cv.imread('img.jpg', 0)


    def filter(img, size_box, type=None):
        img = img[:1200, :1200]  # crop to 700 x 700
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        # create a mask first, center square is 1, remaining all zeros
        if type == 'lowpass':
            mask = np.zeros((rows, cols, 2), np.uint8)
            mask[int(crow - size_box):int(crow + size_box), int(ccol - size_box):int(ccol + size_box)] = 1
        elif type == 'highpass':
            mask = np.ones((rows, cols, 2), np.uint8)
            mask[int(crow - size_box):int(crow + size_box), int(ccol - size_box):int(ccol + size_box)] = 0

        # apply mask and inverse DFT
        fshift = dft_shift * mask
        magnitude_spectrum_mask = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        plt.subplot(221), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks(), plt.yticks()
        plt.subplot(222), plt.imshow(magnitude_spectrum_mask, cmap='gray')
        plt.title('Box Filter ' + str(2 * size_box)), plt.xticks(), plt.yticks()
        plt.subplot(223), plt.imshow(img_back, cmap='gray')
        plt.title(type + ' Filter Output'), plt.xticks(), plt.yticks()

        # Q1B

        if type == 'lowpass':
            f_ishift_slice = np.fft.ifftshift(
                dft_shift[crow - size_box:crow + size_box, ccol - size_box:ccol + size_box, :])
        elif type == 'highpass':
            plt.show()
            return

        img_back = cv2.idft(f_ishift_slice)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        plt.subplot(224), plt.imshow(img_back, cmap='gray')
        plt.title('Sliced Lowpass Filter Output'), plt.xticks(), plt.yticks()
        plt.show()


    # Results
    for i in [10, 25, 50, 100]:
        filter(img, i, 'lowpass')
    for i in [10, 25, 50, 100]:
        filter(img, i, 'highpass')

    # Q2+3

    img = cv.imread('img.jpg', 0)
    img = img[:1201, :1201]  # crop to 700 x 700
    kernel = cv2.getGaussianKernel(151, 20)
    kernel_151_151 = np.dot(kernel, kernel.reshape(1, kernel.shape[0]))
    kernel_1201_1201 = np.zeros((1201, 1201, 2), np.float64)
    x_offset = 524
    y_offset = 524
    kernel_1201_1201[x_offset:kernel.shape[0] + x_offset, y_offset:kernel.shape[0] + y_offset, :] = np.stack(
        (kernel_151_151, kernel_151_151), axis=2)

    # Normal Convolution
    gaus_result = cv2.filter2D(img, -1, kernel_151_151)


    def plot_iff(img, kernel):
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        result = dft_shift * kernel
        result = np.fft.ifftshift(result)
        result_idft = cv2.idft(result)
        return np.log(cv2.magnitude(result_idft[:, :, 0], result_idft[:, :, 1]))


    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(gaus_result, cmap="gray")
    axes[0, 0].set_title('Convolution with Kernel Size=151 StD=20', fontsize=7)

    # More kernels
    kernel_1201_1201_g = cv2.getGaussianKernel(1201, 20)
    kernel_1201_1201_g = np.dot(kernel_1201_1201_g, kernel_1201_1201_g.reshape(1, kernel_1201_1201_g.shape[0]))
    kernel_1201_1201_g = np.stack((kernel_1201_1201_g, kernel_1201_1201_g), axis=2)

    kernel_1201_1201_g_1 = cv2.getGaussianKernel(1201, 10)
    kernel_1201_1201_g_1 = np.dot(kernel_1201_1201_g_1, kernel_1201_1201_g_1.reshape(1, kernel_1201_1201_g_1.shape[0]))
    kernel_1201_1201_g_1 = np.stack((kernel_1201_1201_g_1, kernel_1201_1201_g_1), axis=2)

    axes[0, 1].imshow(plot_iff(img, kernel_1201_1201_g), cmap="gray")
    axes[0, 1].set_title('Multiply in DFT with Kernel Size=1201 StD=20', fontsize=7)
    axes[1, 0].imshow(plot_iff(img, kernel_1201_1201), cmap="gray")
    axes[1, 0].set_title('Multiply in DFT with Kernel Size=151 StD=20, Zero Paded', fontsize=7)
    axes[1, 1].imshow(plot_iff(img, kernel_1201_1201_g_1), cmap="gray")
    axes[1, 1].set_title('Multiply in DFT with Kernel Size=1201 StD=10', fontsize=7)
    fig.savefig('P1Q3.jpg')
    plt.show()
