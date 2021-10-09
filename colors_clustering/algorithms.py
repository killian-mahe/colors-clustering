import numpy as np
import skimage.io


def image_to_pixelmap(image_path: str) -> np.array:
    """
    Convert a image to a np.array of pixels with RGB components.

    Parameters
    ----------
    image_path : str
        Image file path.

    Returns
    -------
    np.array
    """
    pixel_map = skimage.io.imread(image_path).copy()

    # Remove alpha component if it's a PNG file.
    if image_path[-3::] == "png":
        width, height, _ = pixel_map.shape
        pixel_map.resize((width, height, 3))
    return pixel_map
