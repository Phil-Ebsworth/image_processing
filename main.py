
from matplotlib import pyplot as plt
from color import rgb_split
from utils import load_image, show_image, show_multiple_images


def main():
    image_rgb = load_image(path_to_image='Images/Flower.png')
    show_image(image_rgb, title='image_rgb')

    splits = rgb_split(image_rgb)
    show_multiple_images([image_rgb] + splits)

    # channels = ['red', 'green', 'blue']

    """ check_arrays(
        'Exercise 1b',
        ['gamma-correction'],
        [gamma_correction(np.array([
            0., 0.35111917342151316, 0.5785326090814171, 0.8503349277020302, 0.9532375475512688, 1.
        ]).reshape((1,2,3)))],
        [np.array([0., 0.1, 0.3, 0.7, 0.9, 1.]).reshape((1,2,3))],
    )

    check_arrays(
        'Exercise 1c',
        ['rgb_to_gray'],
        [rgb_to_gray(np.eye(3).reshape((1,3,3)))],
        [np.array([.299, .587, .114]).reshape((1,3,1))],
    )

    image_gray = rgb_to_gray(gamma_correction(image_rgb))
    show_image(image_gray, title='image_gray')
    """
    plt.show()


if __name__ == '__main__':
    main()
