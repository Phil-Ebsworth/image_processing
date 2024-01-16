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
    height,width,colors = image_rgb.shape
    color_seperated_images = np.zeros(shape=(height,width,colors,3))
    for c in range(colors):
        color_seperated_images[:,:,c,c] = image_rgb[:,:,c]
    return [color_seperated_images[:,:,:,_] for _ in range(3)]

def gamma_correction(image_rgb, gamma=2.2):
    """Performs gamma correction on a given image.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).
        gamma: The gamma correction factor.

    Returns:
        An array of the shape (h, w, 3) representing the gamma-corrected image.
    """
    image_rgb[:,:,:] **= gamma
    return image_rgb

def rgb_to_gray(image_rgb):
    """Transforms an image into grayscale using reasonable weighting factors.

    Args:
        image_rgb: An rgb-image represented by a numpy array of the shape (h, w, 3).

    Returns:
        An array of the shape (h, w, 1) representing the grayscaled version of the original image.
    """
    
    # Use weighting factors presented in Lecture "bv-01-Sehen-Farbe.pdf", Slide 45
    weights = np.array([.299, .587, .114])
    height,width,colors = image_rgb.shape
    image_grey = np.zeros(shape=(height,width,1))
    for x in range(height):
        for y in range(width):
            for c in range(colors):
                image_grey[x,y,0] += image_rgb[x,y,c] * weights[c]
    return image_grey
# Your solution ends here