import numpy as np


def rgb_split(image_rgb):
    """Splits an rgb-image into its separate channels.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).

    Returns:
        A list containing three numpy arrays, each one has the shape (h, w, 3).
        Each array represents an image where only one color channel of the original image is preserved.
        The order of the preserved channels is red, green, blue.
    """
    result = []

    for channel_index in range(0, image_rgb.shape[2]):
        channel = np.empty_like(image_rgb)
        channel[:, :, channel_index] = image_rgb[:, :, channel_index]
        result.append(channel)

    return result


def gamma_correction(image_rgb, gamma=2.2):
    """Performs gamma correction on a given image.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).
        gamma: The gamma correction factor.

    Returns:
        An array of the shape (h, w, 3) representing the gamma-corrected image.
    """
    return np.power(image_rgb, gamma)


def rgb_to_gray(image_rgb):
    """Transforms an image into grayscale using reasonable weighting factors.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).

    Returns:
        An array of the shape (h, w, 1) representing the grayscaled version of the original image.
    """

    weights = np.array([.299, .587, .114])

    newshape = (image_rgb.shape[0], image_rgb.shape[1], 1)
    grey_scale = np.zeros(newshape)

    for (weight, channel_index) in zip(weights, range(image_rgb.shape[2])):
        grey_scale += image_rgb[:, :, channel_index].reshape(newshape) * weight

    return grey_scale


