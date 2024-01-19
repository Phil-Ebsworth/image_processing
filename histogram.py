from matplotlib import pyplot as plt
import numpy as np

from color import rgb_to_gray
from utils import load_image, plot_hist, show_image_with_hist


def get_hist(image_gray):
    """Computes the histogram of a grayscaled image.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (256,).
        The i-th histogram entry corresponds to the number of image pixels with luminance i.
    """
    hist = np.zeros(256)

    for elem in image_gray.flatten():
        hist[int(elem * 255)] += 1
    return hist


def max_contrast(image_gray):
    """Rescales an images luminance to maximize its contrast.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (h, w, 1) representing the maximal contrastive version of image_gray.
    """

    img = image_gray.copy()
    min_c = img.min() * 255
    max_c = img.max() * 255
    for x in np.nditer(img, op_flags=['readwrite']):
        x = (x - min_c) * (255 / (max_c - min_c)) / 255
    return img

def accumulate_hist(hist):
    """Accumulates and normalizes a given histogram.

    Args:
        hist: An array of the shape (256,).

    Returns:
        An array of the shape (256,) containing the accumulated and normalized values of hist.
    """
    accumulated_hist = np.zeros_like(hist)

    accumulated_hist[0] = hist[0]
    for i in range(1, accumulated_hist.shape[0]):
        accumulated_hist[i] = accumulated_hist[i - 1] + hist[i]

    return accumulated_hist

def equalize_hist(image_gray, accumulated_hist):
    """Performs histogram equalization.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).
        accumulated_hist: An array of the shape (256,) containing the accumulated histogram of image_gray.

    Returns:
        A numpy array of the shape (h, w, 1) representing the equalized image.
    """
    height, width, color = image_gray.shape
    hist_min = accumulated_hist.min()
    hist_max = accumulated_hist.max()

    equalized_image = np.zeros_like(image_gray)
    """ def converter(x):
        index = int(x*255)
        return (accumulated_hist[index] - hist_min) / (hist_max - hist_min)

    equalized_image = np.vectorize(converter)(image_gray) """
    for i in range(height):
        for j in range(width):
            equalized_image[i,j] = (accumulated_hist[int(image_gray[i,j])*255] - hist_min) / (hist_max - hist_min)
    return equalized_image

def main():
    image_gray = rgb_to_gray(load_image('Images/blueFlower.jpg'))
    hist_gray = get_hist(image_gray)
    show_image_with_hist(image_gray, hist_gray)

    
    image_gray_max_contrast = max_contrast(image_gray)
    hist_gray_max_contrast = get_hist(image_gray_max_contrast)
    show_image_with_hist(image_gray_max_contrast, hist_gray_max_contrast)

    
    hist_accumulated = accumulate_hist(hist_gray)
    plot_hist(hist_accumulated)
    
    image_equalized = equalize_hist(image_gray, hist_accumulated)
    hist_equalized = get_hist(image_equalized)
    show_image_with_hist(image_equalized, hist_equalized)
    plot_hist(accumulate_hist(hist_equalized))
    
    plt.show()
    
if __name__ == '__main__':
    main()