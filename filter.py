import numpy as np


def mean_filter(image, w):
    """Applies mean filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    height, width, chs = image.shape

    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    result = np.zeros_like(image)

    for x in range(w, width+w):
        for y in range(w, height+w):
            cropped_image = image_padded[y-w:y+w+1, x-w:x+w+1]
            mean_pixel = np.mean(cropped_image, axis=(0, 1))
            result[y-w, x-w] = mean_pixel

    return result

def median_filter(image, w):
    """Applies median filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zer-padded to preserve the original resolution.
    """
    height, width, chs = image.shape

    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    result = np.zeros_like(image)

    for x in range(w, width+w):
        for y in range(w, height+w):
            cropped_image = image_padded[y-w:y+w+1, x-w:x+w+1]
            mean_pixel = np.median(cropped_image, axis=(0, 1))
            result[y-w, x-w] = mean_pixel

    return result

def get_gauss_kern_2d(w, sigma):
    """Returns a two-dimensional gauss kernel.

    Args:
        w: A parameter controlling the kernel size.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array with shape (2*w+1, 2*w+1) representing a 2d gauss kernel.
        Note that array's values sum to 1.
    """

    gauss_kern = np.ones((2*w+1, 2*w+1))
    for x in range(-w,w+1):
        for y in range(-w,w+1):
            gauss_kern[x+w,y+w] = 1/(2*np.pi*pow(sigma,2))*pow(np.e, -(pow(x,2)+pow(y,2))/(2*pow(sigma,2)))
    return gauss_kern/gauss_kern.sum()

def gauss_filter(image, w, sigma):
    """Applies gauss filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.
        sigma: The σ-parameter of the gaussian density function representing the standard deviation.

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    height, width, chs = image.shape
    

    # Pad the image corners with zeros to preserve the original resolution.
    image_padded = np.pad(image, pad_width=((w,w), (w,w), (0,0)))
    result = np.zeros_like(image)
    gauss_kern = get_gauss_kern_2d(w, sigma)

    for x in range(w, width+w):
        for y in range(w, height+w):
            cropped_image = image_padded[y-w:y+w+1, x-w:x+w+1]
            rearrenged_crop = np.moveaxis(cropped_image, -1, 0)
            new_pixel = np.multiply(rearrenged_crop, gauss_kern).sum(axis=(1, 2))
            result[y-w, x-w] = new_pixel

    return result

def bilateral_filter(image, w, sigma_d, sigma_r):
    """Applies bilateral filtering to the input image.

    Args:
        image: A numpy array with shape (height, width, channels) representing the imput image.
        w: Defines the patch size 2*w+1 of the filter.
        sigma_d: sigma for the pixel distance
        sigma_r: sigma for the color distance

    Returns:
        A numpy array with shape (height, width, channels) representing the filtered image.
        Note that the input image is zero-padded to preserve the original resolution.
    """
    height, width, chs = image.shape
    
    # Pad the image corners with zeros to preserve the original resolution.

    gauss_kern_d = get_gauss_kern_2d(w, sigma_d)[:,:,None]
    result = np.zeros_like(image)
    # TODO: 
    return result