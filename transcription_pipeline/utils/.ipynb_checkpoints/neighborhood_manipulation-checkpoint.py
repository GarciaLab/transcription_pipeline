import warnings
import numpy as np


def extract_neighborhood(image, coordinates, span):
    """
    Extracts a view of a neighborhood of size `span` (or the largest odd number under
    `span` if `span` is even) centered around the pixel corresponding to position 
    specified by `coordinates` from an input `image` in arbitrary dimensions, with
    `span` and `coordinates` both having size matching the dimensions of `image` and
    specifying the neighborhood in respective axes. This is useful when extracting
    proposed spots from the raw data for Gaussian fitting.

    :param image: Input image to extract neighborhood from.
    :type image: Numpy array
    :param coordinates: Coordinates that locate the pixel in `image` around which to
        extract `neighborhood`.
    :type coordinates: Array-like.
    :param span: Size of neighborhood to extract (rounded in each axis to the largest
        odd number below `span` if even).
    :type span: Array-like.
    :return: Extracted neighborhood in same dimensionality as `image`.
    :rtype: Numpy array.
    """
    pixel_coordinates = np.floor(np.asarray(coordinates)).astype(int)
    pixel_span = np.floor(np.asarray(span) / 2).astype(int)
    coordinates_start = pixel_coordinates - pixel_span
    box_dimensions = pixel_span * 2 + 1
    box_indices = tuple((np.indices(box_dimensions).T + coordinates_start).T)

    try:
        neighborhood = image[box_indices]
    except IndexError:
        neighborhood = None

    return neighborhood, coordinates_start
    

def inject_neighborhood(image, neighborhood, coordinates_start):
    """
    Injects a specified neighborhood of dimensionality matching `image` at a position
    with initial corner specified by `coordinates_start` (i.e. the (0, 0,...) corner
    of `neighborhood` is injected at position `coordinates_start`). The input `image`
    is modified in-place, returning `None`. This is useful for injecting spot masks
    into a blank mask when constructing a new mask from Gaussian fitting parameters
    during spot curation.

    :param image: Input image to inject neighborhood in.
    :type image: Numpy array
    :param coordinates_start: Coordinates that locate the position at which to inject
        `neighborhood`, with the 0-indexed corner of `neighborhood` being injected
        at the pixel specified by this parameter.
    :type coordinates_start: Array-like.
    :param neighborhood: Neighborhood to inject, same dimensionality as `image`.
    :type neighborhood: Numpy array.
    :return: None
    :rtype: None
    """
    box_dimensions = neighborhood.shape
    box_indices = tuple((np.indices(box_dimensions).T + coordinates_start).T)

    try:
        image[box_indices] = neighborhood
    except IndexError:
        warnings.warn("Trying to inject neighborhood outside image boundary.")
        pass

    return None