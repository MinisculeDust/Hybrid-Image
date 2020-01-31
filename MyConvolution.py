import numpy as np
# version 3.4 2019.11.8-9:37

def calculate_convolution(image, kernel):

    # rotating kernel with 180 degrees
    kernel = np.rot90(kernel, 2)

    kernel_heigh = int(np.array(kernel).shape[0])
    kernel_width = int(np.array(kernel).shape[1])

    # set kernel matrix to random int matrix
    if ((kernel_heigh % 2 != 0) & (kernel_width % 2 != 0)):  # make sure that the scale of kernel is odd
        # the scale of result
        conv_heigh = image.shape[0] - kernel.shape[0] + 1
        conv_width = image.shape[1] - kernel.shape[1] + 1
        conv = np.zeros((conv_heigh, conv_width))

        # convolve
        for i in range(int(conv_heigh)):
            for j in range(int(conv_width )):
                result = (image[i:i + kernel_heigh, j:j + kernel_width] * kernel).sum()
                # if(result<0):
                #     resutl = 0
                # elif(result>255):
                #     result = 255
                conv[i][j] = result
    return conv

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    # zero padding
    kernel_half_row = int((kernel.shape[0]-1)/2)
    kernel_half_col = int((kernel.shape[1]-1)/2)

    # judge how many channels
    if len(image.shape) == 3:
        image = np.pad(image, ((kernel_half_row, kernel_half_row), (kernel_half_col, kernel_half_col),(0, 0)), 'constant', constant_values=0)

        # if image.shape[2] == 3 or image.shape[2] == 4:
        # if style is png, there will be four channels, but we just need to use the first three
        # if the style is bmp or jpg, there will be three channels
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]
        result_r = calculate_convolution(image_r, kernel)
        result_g = calculate_convolution(image_g, kernel)
        result_b = calculate_convolution(image_b, kernel)
        result_picture = np.dstack([result_r, result_g, result_b])
    # if the picture is black and white
    elif len(image.shape) == 2:
        image = np.pad(image, ((kernel_half_row, kernel_half_row), (kernel_half_col, kernel_half_col)), 'constant', constant_values=0)
        result_picture = calculate_convolution(image, kernel)

    # returns the convolved image (of the same shape as the input image)
    return result_picture


def fourier_trans(image: np.ndarray, kernel: np.ndarray):
    # make the scale of the kernel as the same as pictures
    # make sure it can work for different sance of pictures
    if (image.shape[0] - kernel.shape[0]) % 2 == 0:
        pad_heigh = np.int(((image.shape[0] - kernel.shape[0])) / 2)
    else:
        pad_heigh = np.int(((image.shape[0] - kernel.shape[0])) / 2) + 1

    if (image.shape[1] - kernel.shape[1]) % 2 == 0:
        pad_width = np.int(((image.shape[1] - kernel.shape[1])) / 2)
    else:
        pad_width = np.int(((image.shape[1] - kernel.shape[1])) / 2) + 1
    pad_heigh_light = np.int(((image.shape[0] - kernel.shape[0])) / 2)
    pad_width_light = np.int(((image.shape[1] - kernel.shape[1])) / 2)
    kernel = np.pad(kernel, ((pad_heigh_light, pad_heigh), (pad_width_light, pad_width)), 'constant', constant_values=0)
    print("kernel.shape", kernel.shape)
    copy_fft2_image = np.zeros(image.shape)

    # fourier transform for kernel
    # shift the centre of kernel to axis origin and then do fourier transform
    fft2_kenel_after = np.fft.fft2(np.fft.fftshift(kernel))

    if len(image.shape) == 3:
        # fourier transform
        for i in range(image.shape[2]):
            image_fft = np.fft.fft2(image[:, :, i])
            # print("fft2_kenel_after * image_fft.shape ==== ", (fft2_kenel_after * image_fft).shape)
            # image_fft = np.fft.fftshift(np.fft.fft2(image[:, :, i]))
            # copy_fft2_image[:, :, i] = np.fft.ifftshift(np.fft.ifft2(fft2_kenel_after * image_fft))
            frequcy_result = fft2_kenel_after * image_fft
            # copy_fft2_image[:, :, i] = np.fft.fftshift(np.fft.ifft2(frequcy_result))
            copy_fft2_image[:, :, i] = np.fft.ifft2(frequcy_result)
    elif len(image.shape) == 2:
        image_fft = np.fft.fft2(image)
        frequcy_result = fft2_kenel_after * image_fft
        copy_fft2_image[:, :] = np.fft.ifft2(frequcy_result)

    return copy_fft2_image




