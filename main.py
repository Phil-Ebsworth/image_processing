
from matplotlib import pyplot as plt
import numpy as np
from color import rgb_split, rgb_to_gray
from discreteFourierTransformation import dft_1d, dft_1d_denoise
from filter import box_filter_1d_freq, box_filter_1d_time, gauss_filter, get_gauss_kern_2d, ideal_high_pass_filter, ideal_low_pass_filter, mean_filter, median_filter
from histogram import accumulate_hist, equalize_hist, get_hist, max_contrast
from utils import load_image, plot_hist, plot_time_and_freq, show_image, show_image_with_hist, show_multiple_images, show_time_and_freq_2d


def main():
    """ image_rgb = load_image(path_to_image='Images/Flower.png')
    show_image(image_rgb, title='image_rgb')

    splits = rgb_split(image_rgb)
    show_multiple_images([image_rgb] + splits) """

    # channels = ['red', 'green', 'blue']
    
    """You may vary the parameters w and sigma to explore the effect on the resulting filtered images.
    
    Note: The test-cases will only pass with w=2 and sigma=1.5.
    """
    img = load_image('Images/blueFlower.jpg')
    
    """ # mean filter
    img_mean_filtered = mean_filter(img, w=2)
    
    # median filter
    img_median_filtered = median_filter(img, w=2)
    

    # gauss kern
    gauss_kern = get_gauss_kern_2d(w=2, sigma=1.5)

    # gauss filter
    img_gauss_filtered = gauss_filter(img, w=2, sigma=1.5)
    
    show_image(img, title='original')
    show_image(img_mean_filtered, title='mean filtered')
    show_image(img_median_filtered, title='median filtered')
    show_image(img_gauss_filtered, title='gauss filtered') """
    """Do not change this function at all."""
    
    """ t = np.linspace(0,2*np.pi, 128, dtype=np.complex128)[:-1]
    y_time = np.sin(2*t) + np.cos(11*t) + 1
    y_freq = dft_1d(y_time)
    plot_time_and_freq(t, y_time, y_freq, 'Original Signal')
    
    rng = np.random.default_rng(313373)

    y_time_noisy = y_time + 0.5 * rng.standard_normal(y_time.shape)
    y_freq_noisy = dft_1d(y_time_noisy)
    plot_time_and_freq(t, y_time_noisy, y_freq_noisy, 'Noisy Signal')
    
    y_time_denoised = dft_1d_denoise(y_time_noisy, threshold=0.25)
    y_freq_denoised = dft_1d(y_time_denoised)
    plot_time_and_freq(t, y_time_denoised, y_freq_denoised, 'Denoised Signal')
    
    y_filtered1_time = box_filter_1d_time(y_time, w=5)
    y_filtered1_freq = dft_1d(y_filtered1_time)
    plot_time_and_freq(t, y_filtered1_time, y_filtered1_freq, 'Filtered Signal (Time)')
    
    y_filtered2_time = box_filter_1d_freq(y_time, w=5)
    y_filtered2_freq = dft_1d(y_filtered2_time)
    plot_time_and_freq(t, y_filtered2_time, y_filtered2_freq, 'Filtered Signal (Frequency)')

    square_time = np.zeros((128,128))
    square_time[32:96, 32:96] = 1.
    
    square_freq = np.fft.fft2(square_time)
    show_time_and_freq_2d(
        square_time,
        square_freq,
        title='Original Signal'
    )
    
    square_low_time = ideal_low_pass_filter(square_time, max_frequency=12)
    square_low_freq = np.fft.fft2(square_low_time)
    show_time_and_freq_2d(
        square_low_time,
        square_low_freq,
        title='Low Pass Filtered Signal'
    )
    
    square_high_time = ideal_high_pass_filter(square_time, min_frequency=12)
    square_high_freq = np.fft.fft2(square_high_time)
    show_time_and_freq_2d(
        square_high_time,
        square_high_freq,
        title='High Pass Filtered Signal'
    )
    plt.show() """

if __name__ == '__main__':
    main()
