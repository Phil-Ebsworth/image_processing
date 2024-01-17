import numpy as np

from discreteFourierTransformation import dft_1d, idft_1d


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

def box_filter_1d_time(y_time, w):
    """Applies the mean filter of size 2*w+1 to the input signal.
    
    This function uses convolution to apply the mean filter in time domain.
    
    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the filtered signal in time domain.
    """
    # pad signal periodically 
    n = len(y_time)
    y_time_filtered = np.zeros_like(y_time)

    for i in range(n):
        for j in range(-w, w+1):
            y_time_filtered[i] += y_time[(i + j) % n]

    y_time_filtered /= (2 * w + 1)
    return y_time_filtered

def box_filter_1d_freq(y_time, w):
    """Applies the mean filter of size 2*w+1 to the input signal.
    
    This function uses dft and exploits the convolution theorem to apply the mean filter in frequency domain.
    
    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the filtered signal in time domain.
    """
    y_freq = dft_1d(y_time)
    
    for k in range(len(y_freq)):
        if np.abs(k) > w:
            y_freq[k] = 0.0
            
    return idft_1d(y_freq).real

def ideal_low_pass_filter(y_time, max_frequency):
    """Applies an ideal low pass filter to the input signal.
    
    This function uses dft and exploits the convolution theorem to apply the filter in frequency domain.
    

    Args:
        y_time: A numpy array with shape (height, width) representing the imput signal in time domain.
        max_frequency: The maximal frequency to be preserved in the output signal.

    Returns:
        A numpy array with shape (height, width) representing the filtered signal in time domain.
    """
 
    height, width = y_time.shape
    y_freq = np.fft.fftshift(np.fft.fft2(y_time))
    
    y_filtered_freq = y_freq
    
    lp_filter = np.zeros((height, width))
    center_x, center_y = width // 2, height // 2
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if dist <= max_frequency:
                lp_filter[y,x] = 1
    
    y_filtered_freq *= lp_filter
    
    y_filtered_time = np.fft.ifft2(np.fft.ifftshift(y_filtered_freq))
    return y_filtered_time
    
    
    
def ideal_high_pass_filter(y_time, min_frequency):
    """Applies an ideal low pass filter to the input signal.
    
    This function uses dft and exploits the convolution theorem to apply the filter in frequency domain.
    

    Args:
        y_time: A numpy array with shape (height, width) representing the imput signal in time domain.
        min_frequency: The minimal frequency to be preserved in the output signal.

    Returns:
        A numpy array with shape (height, width) representing the filtered signal in time domain.
    """
    
    height, width = y_time.shape
    y_freq = np.fft.fftshift(np.fft.fft2(y_time))
    y_filtered_freq = y_freq # TODO: Exercise 7b)

    hp_filter = np.zeros((height, width))
    center_x, center_y = width // 2, height // 2
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if dist >= min_frequency:
                hp_filter[y,x] = 1
    
    y_filtered_freq *= hp_filter
    
    y_filtered_time = np.fft.ifft2(np.fft.ifftshift(y_filtered_freq))
    return y_filtered_time