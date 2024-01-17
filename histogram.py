import numpy as np


def get_hist(image_gray):
    """Computes the histogram of a grayscaled image.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (256,).
        The i-th histogram entry corresponds to the number of image pixels with luminance i.
    """
    hist = np.zeros(256)

    for x in range(image_gray.shape[0]):
        for y in range(image_gray.shape[1]):
            hist[int(image_gray[x, y, 0] * 255)] += 1
    return hist


def max_contrast(image_gray):
    """Rescales an images luminance to maximize its contrast.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).

    Returns:
        An array of the shape (h, w, 1) representing the maximal contrastive version of image_gray.
    """

    height, width, colors = image_gray.shape
    min_c = image_gray.min() * 255
    max_c = image_gray.max() * 255
    for x in range(height):
        for y in range(width):
            value = image_gray[x,y,0] * 255
            image_gray[x,y] = (value-min_c)*(255/(max_c - min_c)) / 255
    return image_gray

def accumulate_hist(hist):
    """Accumulates and normalizes a given histogram.

    Args:
        hist: An array of the shape (256,).

    Returns:
        An array of the shape (256,) containing the accumulated and normalized values of hist.
    """
    accumulated_hist = np.zeros_like(hist)

    accumulated_hist[0] = hist[0]
    for index in range(1, hist.size):
        accumulated_hist[index] = accumulated_hist[index - 1] + hist[index]
    accumulated_hist = accumulated_hist / hist.sum()

    return accumulated_hist

def equalize_hist(image_gray, accumulated_hist):
    """Performs histogram equalization.

    Args:
        image_gray: A grayscaled image represented by a numpy array of the shape (h, w, 1).
        accumulated_hist: An array of the shape (256,) containing the accumulated histogram of image_gray.

    Returns:
        A numpy array of the shape (h, w, 1) representing the equalized image.
    """
    hist_min = accumulated_hist.min()
    hist_max = accumulated_hist.max()

    def converter(x):
        index = int(x*255)
        return (accumulated_hist[index] - hist_min) / (hist_max - hist_min)

    equalized_image = np.vectorize(converter)(image_gray)

    return equalized_image