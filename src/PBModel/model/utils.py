import Image
import numpy as np

def loadImage(addr, dim):
    """
    Given address, load image as numpy array of given dimension
    @returns
        numpy array of shape (dim, 3)
    """
    im = Image.open(addr)
    im = im.resize(dim)
    im = im.convert(mode="RGB")
    return np.array(im)
