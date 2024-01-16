import sys
from matplotlib import pyplot as plt
from color import rgb_to_gray
from utils import load_image, show_image


def show_grey_image(img_path = 'Images/Flower.png'):
    image_rgb = load_image(path_to_image=img_path)
    image_grey = rgb_to_gray(image_rgb)
    show_image(image_grey, title='image_grey')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) >1:
        args = globals()
        for i in range(1,len(sys.argv)):
            show_grey_image(sys.argv[i])
    else:
        show_grey_image()