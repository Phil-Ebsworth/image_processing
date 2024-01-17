import numpy as np
from utils import plot_time_and_freq

# Your solution starts here.
def dft_1d(y_time):
    """Transforms a given signal from time to frequency domain.

    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the signal in frequency domain.
    """
    n, = y_time.shape
    
    y_freq = np.zeros_like(y_time) # TODO: Exercise 6a)
    for k in range(n):
        for t in range(n):
            y_freq[k] += y_time[t] * np.exp(-2j * np.pi * k * t / n)
    return y_freq

def idft_1d(y_freq):
    """Transforms a given signal from frequency to time domain.

    Args:
        y_freq: A numpy array with shape (n,) representing the imput signal in frequency domain.

    Returns:
        A numpy array with shape (n,) representing the signal in time domain.
    """
    n, = y_freq.shape
    
    y_time = np.zeros_like(y_freq) # TODO: Exercise 6b)
    for t in range(n):
        for k in range(n):
            y_time[t] += y_freq[k] * np.exp(2j * np.pi * k * t / n) / n
    return y_time

def dft_1d_denoise(y_time, threshold=0.25):
    """Applies a threshold to filter out frequencies with low amplitudes.
    
    Args:
        y_time: A numpy array with shape (n,) representing the imput signal in time domain.

    Returns:
        A numpy array with shape (n,) representing the denoised signal in time domain.
    """
    n, = y_time.shape
    y_freq = dft_1d(y_time) 
    for k in range(n):
        if (2 / n * np.abs(y_freq[k])) < threshold:
            y_freq[k] = 0.0
    return idft_1d(y_freq)