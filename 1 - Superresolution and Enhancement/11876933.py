'''
Name: Gianluca Capezzuto Sardinha
NUSP: 11876933
Course Code: SCC0651
Year/Semester: 2024/1
Title: Enhancement and Superresolution
'''

import numpy as np
import imageio.v3 as iio

def rmse(enh_img, ref_img):
    '''
    Calculate the root mean square error between two images.
    enh_img: enhanced image.
    ref_img: reference image.
    '''
    return round(np.sqrt(np.mean((enh_img.astype(np.int16) - ref_img.astype(np.int16)) ** 2)), 4)

def load_images(name):
    '''
    Find and load to a list all low resolution images.
    directory: path to the directory containing the images (Initially, the function was receiving 
    the directory path as an argument, but I changed to run the code in the run.codes).
    name: name of the image
    '''
    imgs = []

    for type in range(0, 4):
        img = iio.imread(str(name) + str(type) + '.png')
        imgs.append(img)

    return imgs

def map_indices(i, j):
    '''
    Map the indices of the low resolution images to the high resolution image.
    i: row index of the image.
    j: column index of the image.
    '''
    if i % 2 == 0 and j % 2 == 0:
        return 0
    elif i % 2 == 1 and j % 2 == 0:
        return 1
    elif i % 2 == 0 and j % 2 == 1:
        return 2
    elif i % 2 == 1 and j % 2 == 1:
        return 3

def superresolution(low_imgs, high_img):
    '''
    Upscale the low resolution images to the same size as the high resolution image.
    low_imgs: list of low resolution images.
    high_img: high resolution image.
    '''
    h, w = high_img.shape
    # New image with the same size as the high resolution image, waiting to be filled with the upscaled low resolution images
    low_img_upscaled = np.zeros((h, w)) 

    # Upscale the low resolution images
    for i in range(0, h):
        for j in range(0, w):
            low_img_upscaled[i, j] = low_imgs[map_indices(i, j)][i // 2, j // 2]

    low_img_upscaled = low_img_upscaled.astype(np.uint8)
    # iio.imwrite('output/' + imglow + '_upscaled.png', low_img_upscaled)

    return rmse(low_img_upscaled, high_img)

def histogram(img, no_levels = 256):
    '''
    Calculate the histogram of an image.
    img: input image.
    no_levels: number of levels in the histogram.
    '''
    # Initialize the histogram
    hist = np.zeros(no_levels).astype(int)

    # For each pixel value, count the number of pixels with that value
    for i in range(no_levels):
        no_pixel_value_i = np.sum(img == i)
    
        hist[i] = no_pixel_value_i

    return hist

def single_cumulative_histogram(imgs, no_levels = 256):
    '''
    Compute the single cumulative histogram of each image.
    imgs: list of images.
    no_levels: number of levels in the histogram.
    '''
    hist_transf_list = []
    img_eq_list = []

    for img in imgs:
        # Get the histogram of each image
        hist = histogram(img, no_levels)

        # Initialize the cumulative histogram
        histC = np.zeros(no_levels).astype(int)
        histC[0] = hist[0]

        # Calculate the cumulative histogram
        for i in range(1, no_levels):
            histC[i] = hist[i] + histC[i - 1]

        # Transfer function
        hist_transf = np.zeros(no_levels).astype(np.uint8)

        N, M = img.shape
        img_eq = np.zeros([N, M]).astype(np.uint8)

        # Calculate the transfer function
        for z in range(no_levels):
            s = ((no_levels - 1) / float(M * N)) * histC[z]
            hist_transf[z] = s

            # For every coordinate in which A == z, replace it with s
            img_eq[np.where(img == z)] = s

        hist_transf_list.append(hist_transf)
        img_eq_list.append(img_eq)

    return img_eq_list, hist_transf_list

def joint_cumulative_histogram(imgs, no_levels = 256):
    '''
    Compute the joint cumulative histogram of the images
    imgs: list of images.
    no_levels: number of levels in the histogram.
    '''
    hist_images = []    
    hist_transf_list = []
    img_eq_list = []

    for img in imgs:
        # Get the histogram of each image
        hist = histogram(img, no_levels)

        # Initialize the cumulative histogram
        histC = np.zeros(no_levels).astype(int)
        histC[0] = hist[0]

        # Calculate the cumulative histogram
        for i in range(1, no_levels):
            histC[i] = hist[i] + histC[i - 1]

        # Save the cumulative histogram of each image
        hist_images.append(histC)

    # Initialize and calculate the joint cumulative histogram
    histC = np.zeros(no_levels).astype(int)

    for i in range(0, 4):
        histC += hist_images[i]

    for img in imgs:
        # Transfer function
        hist_transf = np.zeros(no_levels).astype(np.uint8)

        N, M = img.shape
        img_eq = np.zeros([N, M]).astype(np.uint8)

        # Calculate the transfer function
        for z in range(no_levels):
            # The formula was modified considering the number of images in the histogram list, 
            # so with that we normalize the histogram values.
            s = ((1/4) * (no_levels - 1) / float(M * N)) * histC[z]
            hist_transf[z] = s

            # For every coordinate in which A == z, replace it with s
            img_eq[np.where(img == z)] = s

        hist_transf_list.append(hist_transf)
        img_eq_list.append(img_eq)

    return img_eq_list, hist_transf_list

def gamma_correction(imgs, gamma):
    '''
    Apply gamma correction to the low resolution images.
    imgs: list of images.
    gamma: gamma value.
    '''
    img_eq_list = []

    # Apply gamma correction to each image
    for img in imgs:
        # Function defined by the formula: s = 255 * ((r / 255) ** (1 / gamma))
        img_eq = (255 * ((img / 255) ** (1 / gamma))).astype(np.uint8) 
        img_eq_list.append(img_eq)

    return img_eq_list

if __name__ == "__main__":
    imglow = input().rstrip()
    imghigh = input().rstrip()
    f = int(input().rstrip())
    gamma = float(input().rstrip())

    # Load the low resolution images and the high resolution image
    low_imgs = load_images(imglow)
    imghigh = iio.imread(imghigh)

    # Apply the selected method. In the local tests, I was saving the images in the output directory 
    # to check the results, but I removed this part to run the code in the run.codes.
    if f == 0:
        error = superresolution(low_imgs, imghigh)

        print(error)
    elif f == 1:
        enhanced_imgs, _ = single_cumulative_histogram(low_imgs)

        error = superresolution(enhanced_imgs, imghigh)

        print(error)
    elif f == 2:
        enhanced_imgs, _ = joint_cumulative_histogram(low_imgs)

        error = superresolution(enhanced_imgs, imghigh)

        print(error)
    elif f == 3:
        enhanced_imgs = gamma_correction(low_imgs, gamma)

        error = superresolution(enhanced_imgs, imghigh)

        print(error)
