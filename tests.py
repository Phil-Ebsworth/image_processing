from color import rgb_split


def test_rgb_split():
    splits = rgb_split("Images/Flower.png")
    assert (splits) == rgb_split("Images/Flower.png")
