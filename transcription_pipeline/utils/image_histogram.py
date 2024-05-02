import numpy as np


"""
This utility module copies in the code for some private functions in the `skimage`
library useful for constructing image histograms. We are copying in these functions
for forward-compatibility in case we want to run the pipeline on later versions of
`skimage` since private function APIs are not guaranteed to remain unchanged. The
code that follows is copied from `skimage.exposure` with minimal modifications to
better fit our parallelization workflow and to remove channel axis handling for
simplification.
"""


def _bincount_histogram_centers(hist_range):
    """
    Compute bin centers for bincount-based histogram.
    """
    image_min, image_max = hist_range
    image_min = int(image_min.astype(np.int64))
    image_max = int(image_max.astype(np.int64))

    bin_centers = np.arange(image_min, image_max + 1)

    return bin_centers


def _offset_array(arr, low_boundary, high_boundary):
    """Offset the array to get the lowest value at 0 if negative."""
    if low_boundary < 0:
        offset = low_boundary
        dyn_range = high_boundary - low_boundary
        # get smallest dtype that can hold both minimum and offset maximum
        offset_dtype = np.promote_types(
            np.min_scalar_type(dyn_range), np.min_scalar_type(low_boundary)
        )
        if arr.dtype != offset_dtype:
            # prevent overflow errors when offsetting
            arr = arr.astype(offset_dtype)
        arr = arr - offset
    return arr


def _bincount_histogram(image, hist_range, bin_centers=None):
    """
    Efficient histogram calculation for an image of integers.

    This function is significantly more efficient than np.histogram but
    works only on images of integers. It is based on np.bincount.

    :param image: Input image.
    :type image: np.ndarray
    :param hist_range: Range of values covered by the histogram bins.
    :type hist_range: 2-tuple of int
    :return: The values of the histogram and the values at the center of the bins.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if bin_centers is None:
        bin_centers = _bincount_histogram_centers(hist_range)
    image_min, image_max = bin_centers[0], bin_centers[-1]
    image = _offset_array(image, image_min, image_max)
    # noinspection PyTypeChecker
    hist = np.bincount(image.ravel(), minlength=image_max - min(image_min, 0) + 1)

    # noinspection PyTypeChecker
    idx = max(image_min, 0)
    hist = hist[idx:]
    return hist, bin_centers


def _get_outer_edges(image, hist_range):
    """
    Determine the outer bin edges to use for `numpy.histogram`, obtained from
    either the image or hist_range.
    """
    if hist_range is not None:
        first_edge, last_edge = hist_range
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in hist_range parameter.")
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"supplied hist_range of [{first_edge}, {last_edge}] is " f"not finite"
            )
    elif image.size == 0:
        # handle empty arrays. Can't determine hist_range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = image.min(), image.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"autodetected hist_range of [{first_edge}, {last_edge}] is "
                f"not finite"
            )

    # expand empty hist_range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(image, nbins, hist_range):
    """
    Computes histogram bins for use with `numpy.histogram`.
    """
    first_edge, last_edge = _get_outer_edges(image, hist_range)
    # numpy/gh-10322 means that type resolution rules are dependent on array
    # shapes. To avoid this causing problems, we pick a type now and stick
    # with it throughout.
    bin_type = np.result_type(first_edge, last_edge, image)
    if np.issubdtype(bin_type, np.integer):
        bin_type = np.result_type(bin_type, float)

    # compute bin edges
    bin_edges = np.linspace(
        first_edge, last_edge, nbins + 1, endpoint=True, dtype=bin_type
    )
    return bin_edges


def histogram(image, bins, hist_range, normalize):
    """
    Return histogram of image.

    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.

    :param image: Image for which the histogram is to be computed.
    :type image: np.ndarray
    :param bins: The number of histogram bins. For images with integer dtype, an array
        containing the bin centers can also be provided. For images with floating point
        dtype, this can be an array of bin_edges for use by `np.histogram`.
    :type bins: {int, np.ndarray}
    :param hist_range: Range of values covered by the histogram bins.
    :type hist_range: 2-tuple of scalars
    :param bool normalize: If True, normalize the histogram by the sum of its values.
    """

    image = image.flatten()
    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        bin_centers = bins if isinstance(bins, np.ndarray) else None
        hist, bin_centers = _bincount_histogram(image, hist_range, bin_centers)
    else:
        hist, bin_edges = np.histogram(image, bins=bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers
