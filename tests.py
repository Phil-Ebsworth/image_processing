from color import rgb_split
from utils import load_image


def test_rgb_split():
    img = load_image(path_to_image='Images/Flower.png')
    splits = rgb_split(img)
    assert (splits) == rgb_split(img)
