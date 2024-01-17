
from matplotlib import pyplot as plt
from color import rgb_split, rgb_to_gray
from filter import gauss_filter, get_gauss_kern_2d, mean_filter, median_filter
from histogram import accumulate_hist, equalize_hist, get_hist, max_contrast
from utils import load_image, plot_hist, show_image, show_image_with_hist, show_multiple_images


def main():
    """ image_rgb = load_image(path_to_image='Images/Flower.png')
    show_image(image_rgb, title='image_rgb')

    splits = rgb_split(image_rgb)
    show_multiple_images([image_rgb] + splits) """

    # channels = ['red', 'green', 'blue']

    """ image_gray = rgb_to_gray(load_image('Images/blueFlower.jpg'))
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
    plot_hist(accumulate_hist(hist_equalized)) """

    
    
    """You may vary the parameters w and sigma to explore the effect on the resulting filtered images.
    
    Note: The test-cases will only pass with w=2 and sigma=1.5.
    """
    img = load_image('Images/blueFlower.jpg')
    
    # mean filter
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
    show_image(img_gauss_filtered, title='gauss filtered')
    
    plt.show()

if __name__ == '__main__':
    main()
