import sys

from matplotlib import pyplot as plt
from color import rgb_split
from utils import load_image, show_image, show_multiple_images


def show_rgb_split(img_path = 'Images/Flower.png'):
    image_rgb = load_image(path_to_image=img_path)
    splits = rgb_split(image_rgb)
    show_multiple_images([image_rgb] + splits)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = globals()
        for i in range(1,len(sys.argv)):
            show_rgb_split(sys.argv[i])
    else:
        show_rgb_split()