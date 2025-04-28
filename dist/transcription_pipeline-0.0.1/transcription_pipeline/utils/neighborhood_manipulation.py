import warnings
import numpy as np


def ellipsoid(diameter, height):
    """
    Constructs an ellipsoid footprint for morphological operations - this is usually
    better than built-in skimage.morphology footprints because the voxel dimensions
    in our images are typically anisotropic.

    :param int diameter: Diameter in xy-plane of ellipsoid footprint.
    :param int height: Height in z-axis of ellipsoid footprint.
    :return: Ellipsoid footprint.
    :rtype: bool
    """
    # Coerce diameter and height to odd integers (skimage requires footprints to be
    # odd in size).
    if diameter < 3 or height < 3:
        raise Exception(
            " ".join(
                [
                    "Setting diameter or height below 3 results in an",
                    "empty or improperly dimensioned footprint.",
                ]
            )
        )

    def round_odd(num):
        """
        Helper function to round to the nearest odd integer.
        """
        return int(((num + 1) // 2) * 2 - 1)

    diameter = round_odd(diameter)
    height = round_odd(height)

    # Generate coordinate arrays
    x = np.arange(-diameter // 2 + 1, diameter // 2 + 1)
    y = np.copy(x)
    z = np.arange(-height // 2 + 1, height // 2 + 1)

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    ellipsoid_eqn = (
        (xx / (diameter / 2)) ** 2
        + (yy / (diameter / 2)) ** 2
        + (zz / (height / 2)) ** 2
    )
    ellipsoid_footprint = ellipsoid_eqn < 1

    return ellipsoid_footprint


def extract_neighborhood(image, coordinates, span):
    """
    Extracts a view of a neighborhood of size `span` (or the largest odd number under
    `span` if `span` is even) centered around the pixel corresponding to position
    specified by `coordinates` from an input `image` in arbitrary dimensions, with
    `span` and `coordinates` both having size matching the dimensions of `image` and
    specifying the neighborhood in respective axes. This is useful when extracting
    proposed spots from the raw data for Gaussian fitting.

    :param image: Input image to extract neighborhood from.
    :type image: np.ndarray
    :param coordinates: Coordinates that locate the pixel in `image` around which to
        extract `neighborhood`.
    :type coordinates: np.ndarray
    :param span: Size of neighborhood to extract (rounded in each axis to the largest
        odd number below `span` if even).
    :type span: np.ndarray
    :return: Extracted neighborhood in same dimensionality as `image`.
    :rtype: np.ndarray
    """
    pixel_coordinates = np.floor(np.asarray(coordinates)).astype(int)
    pixel_span = np.floor(np.asarray(span) / 2).astype(int)
    coordinates_start = pixel_coordinates - pixel_span

    if np.all(coordinates_start >= 0):
        box_dimensions = pixel_span * 2 + 1
        box_indices = tuple((np.indices(box_dimensions).T + coordinates_start).T)

        try:
            neighborhood = image[box_indices]
            # The NaN handling is done for padded or masked movies.
            # `.sum()` is faster than `.any` on multidimensional arrays.
            if np.isnan(neighborhood).sum():
                neighborhood = None
        except IndexError:
            neighborhood = None
    else:
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
    :type image: np.ndarray
    :param coordinates_start: Coordinates that locate the position at which to inject
        `neighborhood`, with the 0-indexed corner of `neighborhood` being injected
        at the pixel specified by this parameter.
    :type coordinates_start: {np.ndarray, int}
    :param neighborhood: Neighborhood to inject, same dimensionality as `image`.
    :type neighborhood: np.ndarray
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
