import warnings
import numpy as np
from skimage.filters import (
    difference_of_gaussians,
    rank,
    gaussian,
    threshold_otsu,
    threshold_li,
    sobel,
)
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_objects,
    binary_dilation,
)
from skimage.util import img_as_ubyte, img_as_float32
from scipy import ndimage as ndi
from functools import partial
from ..utils import parallel_computing


def _determine_num_iter(footprint):
    """
    Parses a footprint parameter passed as a boolean numpy array (applied once),
    as a tuple specifying the number of iterations and the boolean array setting
    the footprint, or as a nested tuple specifying the range of number of iterations
    to try when determining the optimal number of iterations as a tuple and a
    boolean numpy array.
    """
    # Initialize num_iter_range to zero so that no iterations are recorded and used
    # to automatically determine a number of iterations
    num_iter_range = 1

    # Check footprint type to decide number of iterations
    if isinstance(footprint, tuple):
        iter_range = footprint[0]
        footprint = footprint[1]
        if isinstance(iter_range, tuple):
            num_iter_lower = iter_range[0]
            num_iter_upper = iter_range[1]

            num_iter = num_iter_lower - 1
            num_iter_range = num_iter_upper - num_iter

        elif isinstance(iter_range, int):
            num_iter = iter_range

        else:
            raise TypeError(
                " ".join(
                    [
                        "Number of iterations in footprint parameter",
                        "must be given as int or 2-tuple of int.",
                    ]
                )
            )

    elif isinstance(footprint, np.ndarray):
        num_iter = 1
    else:
        raise TypeError("footprint parameter must being either a tuple or numpy array.")

    return footprint, num_iter, num_iter_range


def _iterate_local_max(image, mask, iteration_params):
    """
    Uses a series of maximum dilations compared to the original image to select
    local maxima inside a provided mask as per the iteration_params as returned by
    :func:`_determine_num_iter`.
    """
    footprint, num_iter, num_iter_range = iteration_params

    # We apply a maximum dilation to the image, then compare to the original image
    # such that the only points that are selected correspond to maxima within the net
    # footprint of the dilation after the iterated application of the max filter.
    image_max = np.copy(image) * mask  # Remove spurious peaks in background
    for _ in range(num_iter):
        image_max = ndi.maximum_filter(image_max, footprint=footprint)

    # Perform iterative max dilation in specified range and record results
    max_dilations_iter = np.empty((num_iter_range,) + image.shape, dtype=image.dtype)
    max_dilations_iter[0] = image_max
    for i in range(1, num_iter_range):
        max_dilations_iter[i] = ndi.maximum_filter(
            max_dilations_iter[i - 1], footprint=footprint
        )

    # Construct boolean mask marking local maxima for each number of iterations
    peak_mask_iter = np.empty(max_dilations_iter.shape, dtype=bool)
    for i in range(peak_mask_iter.shape[0]):
        peak_mask_iter[i] = max_dilations_iter[i] == image

    return peak_mask_iter


def _find_plateau(array, **kwargs):
    """
    Find the center of the largest plateau in a monotonically decreasing array of
    integers, with a plateau being defines as contiguous stretches of values
    differing by less than max_diff. Defaults to minimizing successive differences
    if no plateau is found, where the averaging window before minimization can
    be set by an `averaging_window` kwarg.
    """
    try:
        max_diff = kwargs["max_diff"]
    except KeyError:
        max_diff = 1

    plateau = np.zeros((array.size, array.size), dtype=bool)

    # This suffices to find all plateaus because the input array is monotonically
    # decreasing such that points within some tolerance max_diff of any given
    # value are necessarily contiguous.
    for i in range(array.size):
        plateau[i] = np.abs(array - array[i]) <= max_diff

    plateau_size = np.sum(plateau, axis=1)

    # Check if any plateaus were found
    num_plateaus = np.sum(plateau_size > 1)

    # If no plateaus, default to minimizing successive differences
    if num_plateaus > 0:
        plateau_index = np.argmax(plateau_size)
        plateau_start = np.argmax(plateau[plateau_index])
        stationary_point = plateau_start + np.floor(
            plateau_size[plateau_index] / 2
        ).astype(int)

    else:
        warnings.warn(
            "".join(
                [
                    "No plateau found with specified max_differences, ",
                    "defaulting to minimizing successive differences.",
                ]
            )
        )

        # Compute successive differences between number of peaks for range of iterations
        diff_array = [s - t for s, t in zip(array, array[1:])]
        diff_array = np.array(diff_array, dtype=float)

        # Perform rolling average
        try:
            averaging_window = kwargs["averaging_window"]
        except KeyError:
            averaging_window = 3

        diff_array = ndi.uniform_filter1d(diff_array, size=averaging_window)

        stationary_point = np.argmin(diff_array)

    return stationary_point


def _iterative_peak_warnings(
    stationary_point, num_iter_range, frame_message, frame_message_end
):
    """
    Constructs warnings depending on value of stationary point found during
    optimization of footprint iteration for peak finding.
    """
    if stationary_point == 0:
        warnings.warn(
            "".join(
                [
                    frame_message,
                    "Stationary point is at lower bound given for ",
                    "number of iterations, consider extending ",
                    "range.",
                    frame_message_end,
                ]
            ),
            stacklevel=2,
        )
    if stationary_point == (num_iter_range - 1):
        warnings.warn(
            "".join(
                [
                    frame_message,
                    "Stationary point is at upper bound given for ",
                    "number of iterations, consider extending ",
                    "range.",
                    frame_message_end,
                ]
            ),
            stacklevel=2,
        )

    return None


# noinspection PyIncorrectDocstring
def iterative_peak_local_max(image, footprint, *, mask, frame_index, **kwargs):
    """
    Find peaks in an image as coordinate list.

    :param image: 2D (projected) or 3D image of a nuclear marker (usually used
        on pre-processed images, for instance with a DoG filter).
    :type image: np.ndarray
    :param footprint: Footprint used during maximum dilation. This sets the minimum
        distance between peaks, with a single maximum within the net footprint of
        iterated maximum dilations. Can be given as a Numpy array of booleans that
        gets used as a footprint for a single maximum dilation, as a
        tuple(num_iter, footprint) where the footprint is used for maximum diation
        num_iter times, as tuple((num_iter_lower, num_iter_upper), footprint)
        where num_iter_lower and num_iter_upper are the lowest and highest number of
        iterations respectively that the function will try when looking for a
        stationary point in the number of detected maxima for each frame so that no
        frame is over- or under-segmented.
    :type footprint: {np.ndarray, tuple[int, np.ndarray], tuple[tuple[int, int], np.ndarray]}
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations).
    :type mask: np.ndarray
    :param frame_index: Index of frame being processed, used to make warning
        about footprint more descriptive.
    :type frame_index: int
    :param int max_diff: Maximum difference allowed for contiguous points to be
        considered a plateau when looking for a stationary value of the number
        of detected nuclei with respect to the distance between peaks as set by
        the number of iterations of the footprint. Defaults to 1.
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection. Defaults to 3, only used as fallback if plateau finding fails.
    :return: Boolean mask labeling local maxima.
    :rtype: np.ndarray
    """
    # Determine iteration parameters
    iteration_params = _determine_num_iter(footprint)

    # Find local maxima for specified range of iterations
    peak_mask_iter = _iterate_local_max(image, mask, iteration_params)

    # Count number of marked maxima for each number of iterations by summing
    # boolean mask of peaks over all but first axis (iteration axis)
    sum_axes = tuple(range(peak_mask_iter.ndim))[1:]
    num_peaks_iter = np.sum(peak_mask_iter, axis=sum_axes)

    # Check if frame index should be referenced by warnings if necessary
    if frame_index is not None:
        frame_message = "Frame {0}: ".format(frame_index)
        frame_message_end = (
            " Note: Frame number inaccurate if chunking and parallelizing."
        )
    else:
        frame_message = ""
        frame_message_end = ""

    num_iter_range = iteration_params[2]
    stationary_point = 0
    if num_iter_range > 1:  # Checks if iterations were requested
        stationary_point = _find_plateau(num_peaks_iter, **kwargs)
        _iterative_peak_warnings(
            stationary_point, num_iter_range, frame_message, frame_message_end
        )

    peak_mask = peak_mask_iter[stationary_point]

    return peak_mask


# noinspection PyIncorrectDocstring
def denoise_frame(stack, denoising, **kwargs):
    """
    Denoises a z-stack using specified method (gaussian or median filtering).

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: np.ndarray
    :param denoising: Determines which method to use for initial denoising of the
        image (before any filtering or morphological operations) between a gaussian
        filter and a median filter.

        * ``gaussian``: requires a ``denoising_sigma`` keyword argument to determine
          the sigma parameter for the gaussian filter.
        * ``median``: requires a ``median_footprint`` keyword argument to determine
          the footprint used for the median filter.

    :type denoising: {'gaussian', 'median'}
    :type denoising_sigma: scalar or sequence of scalars, only required if using
        ``denoising='gaussian'``.
    :param median_footprint: Footprint used for median filter denoising of the image
        prior to any morphological operations or other filtering.
    :type median_footprint: np.ndarray[np.bool]
    :return: Denoised stack.
    :rtype: np.ndarray
    """
    # Normalize image
    stack = img_as_float32(stack)
    stack = stack / np.max(np.abs(stack))

    # Denoising step
    if denoising == "gaussian":
        try:
            denoising_sigma = kwargs["denoising_sigma"]
        except KeyError:
            raise Exception("Gaussian denoising requires a denoising_sigma parameter.")
        denoised_stack = gaussian(stack, sigma=denoising_sigma)

    elif denoising == "median":
        try:
            median_footprint = kwargs["median_footprint"]
        except KeyError:
            raise Exception("Median denoising requires a median_footprint parameter.")
        stack = img_as_ubyte(stack)
        denoised_stack = rank.median(stack, footprint=median_footprint)

    else:
        raise Exception("Unrecognized denoising parameter.")

    return denoised_stack


def _find_largest_cc(labelled_mask):
    """
    Extracts a RegionProperties object for the largest connected component.
    """
    measured_labels = regionprops(labelled_mask)
    component_sizes = np.array([component.num_pixels for component in measured_labels])
    largest_component_idx = np.argmax(component_sizes)
    largest_cc_regionprops = measured_labels[largest_component_idx]
    return largest_cc_regionprops


def _check_for_backgound(binarized_mask, min_span):
    """
    Checks a binarized nuclear mask for the presence of unexpectedly large connected
    components that would indicate surface background.
    """
    labelled_mask = label(binarized_mask)
    largest_cc_regionprops = _find_largest_cc(labelled_mask)
    largest_cc_bbox = largest_cc_regionprops.bbox

    # If largest connected component spans a large enough section of the FOV, it is
    # likely surface background
    bbox_bounds = np.split(np.asarray(largest_cc_bbox), 2)
    span = bbox_bounds[1] - bbox_bounds[0]
    has_large_cc = np.all(span > np.asarray(min_span))

    return has_large_cc


def _background_mask(
    frame,
    sigma_blur,
    max_span,
    threshold_method="otsu",
    background_dilation_footprint=None,
):
    """
    Uses a large Gaussian blur to lowpass the image and find large regions of high
    background by Otsu thresholding. A mask of the background is return to help with
    background subtraction downstream.
    """
    # Blur the input frame, preferably with an asymmetric kernel small in z and much
    # larger than the nuclei in xy
    gaussian_blur = gaussian(frame, sigma=sigma_blur)

    # Threshold to find background
    threshold = threshold_otsu(gaussian_blur)

    if threshold_method == "li":
        threshold = threshold_li(gaussian_blur, initial_guess=threshold)
    elif threshold_method == "otsu":
        pass
    else:
        raise ValueError("`threshold_method` parameter not recognized.")

    binarized_background = gaussian_blur > threshold

    # Choose the largest connected component in the blurred image as the likeliest
    # surface background component
    labeled_background = label(binarized_background)
    background_regionprop = _find_largest_cc(labeled_background)

    # We now check the bounding box of the largest connected component to make sure
    # that it is in fact surface noise - otherwise, it could just be a noisy dataset
    # causing the nuclei to join together as a large connected component under
    # Gaussian filtering and binarization.
    background_bbox = background_regionprop.bbox
    bbox_bounds = np.split(np.asarray(background_bbox), 2)
    span = bbox_bounds[1] - bbox_bounds[0]

    # Check if surface noise is at the top or bottom of the z-stack
    if (bbox_bounds[0][0] == 0) or (bbox_bounds[1][0] == frame.shape[0]):
        background_spans_surface = True
    else:
        background_spans_surface = False

    # If connected component from blurred image is not near surface or spans too
    # much of the stack, do not mark as background.
    if background_spans_surface and np.all(span <= max_span):
        # noinspection PyTestUnpassedFixture
        binarized_background = labeled_background == background_regionprop.label
        binary_dilation(
            binarized_background,
            footprint=background_dilation_footprint,
            out=binarized_background,
        )
    else:
        binarized_background = np.zeros_like(binarized_background)

    return binarized_background


# noinspection PyIncorrectDocstring
def binarize_frame(
    stack,
    *,
    thresholding,
    opening_footprint,
    closing_footprint,
    cc_min_span,
    background_max_span,
    background_sigma,
    background_threshold_method,
    background_dilation_footprint,
    **kwargs
):
    """
    Binarizes a z-stack using specified thresholding method, separating between
    foreground and background.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: np.ndarray
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.

        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
          the footprint used for the local Otsu thresholding.

    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param opening_footprint: Footprint used for closing operation.
    :type opening_footprint: np.ndarray
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: np.ndarray
    :param cc_min_span: Minimum span in each axis that the largest connected component
        in a binarized image must have to be considered possible background noise. The
        key here is that large x- and y-axis span (more than a few nuclei) indicates
        either a region of high-intensity surface noise or a low signal-to-noise image
        that causes nuclei to appear connected when binarized. This flags the image for
        further processing and possible background removal.
    :type cc_min_span: np.ndarray
    :param background_max_span: Maximum span in each axis that a connected component
        flagged as surface noise can have and still be consisdered surface noise (i.e.
        if a large part of the stack is surface noise, the data might not be usable
        in the first place and will be very difficult to segment in 3D so we stop the
        analysis). Connected blurred regions with bounding boxes spanning this parameter
        will be removed from consideration when thresholding to obtain a nuclear mask.
    :type background_max_span: np.ndarray
    :param background_sigma: Standard deviation to use in each axis for Gaussian blurring
        of image prior to segmenting out the surface noise. This should be small in z
        so as not to bleed through the z-stack, and much larger than the nuclei in x- and
        y- so as to only keep low-frequency noise.
    :type background_sigma: np.ndarray
    :param background_threshold_method: Method to use for thresholding the Gaussian-
        blurred image for surface noise detection. Only global Otsu and Li methods
        are implemented.
    :type background_threshold_method: {"otsu", "li"}
    :param background_dilation_footprint: Structuring element used for binary dilation
        of the background surface noise mask when removing background noise.
    :type background_dilation_footprint: np.ndarray[np.bool]
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: np.ndarray[np.bool]
    :return: A boolean array of the same type and shape as stack, with True values
        corresponding to the foreground and False to the background.
    :rtype: np.ndarray
    """

    # Thresholding step
    if thresholding == "global_otsu":
        threshold = threshold_otsu(stack)

    elif thresholding == "local_otsu":
        try:
            otsu_footprint = kwargs["otsu_footprint"]
        except KeyError:
            raise Exception(
                "Local Otsu thresholding requires an otsu_footprint parameter."
            )
        # Convert denoised stack to uint8 for rank operation
        stack = img_as_ubyte(stack)
        threshold = rank.otsu(stack, otsu_footprint)

    elif thresholding == "li":
        threshold_guess = threshold_otsu(stack)
        threshold = threshold_li(stack, initial_guess=threshold_guess)

    else:
        raise Exception("Unrecognized thresholding parameter.")

    # Binarize stack by thresholding
    binarized_stack = stack >= threshold

    # Check if thresholded stack has large connected components that would indicate
    # surface background
    has_background = _check_for_backgound(binarized_stack, cc_min_span)

    # If stack has surface background, make a mask of background by blurring
    # heavily in xy and binarizing
    if has_background:
        background_mask = _background_mask(
            stack,
            background_sigma,
            background_max_span,
            background_threshold_method,
            background_dilation_footprint,
        )

        if thresholding == "global_otsu":
            threshold = threshold_otsu(stack[~background_mask])
        elif thresholding == "local_otsu":
            # If thresholding is done using local Otsu, the conversion to ubyte
            # and extraction of the footprint kwarg will have already happened
            # noinspection PyUnboundLocalVariable
            threshold = rank.otsu(stack, otsu_footprint, mask=(~background_mask))
        elif thresholding == "li":
            threshold_guess = threshold_otsu(stack[~background_mask])
            threshold = threshold_li(
                stack[~background_mask], initial_guess=threshold_guess
            )

        binarized_stack = (stack >= threshold) * (~background_mask)

    # Remove remnant surface backgound using an opening operation
    binary_opening(binarized_stack, footprint=opening_footprint, out=binarized_stack)

    # Clean up binarized image with a closing operation
    binary_closing(binarized_stack, footprint=closing_footprint, out=binarized_stack)

    return binarized_stack


# noinspection PyIncorrectDocstring
def mark_frame(stack, mask, *, low_sigma, high_sigma, max_footprint, **kwargs):
    """
    Uses a difference of gaussians bandpass filter to enhance nuclei, then a local
    maximum to find markers for each nucleus. Being permissive with the filtering at
    this stage is recommended, since further filtering of the nuclear localization can
    be done post-segmentation using the size and morphology of the segmented objects.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: np.ndarray
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations).
    :type mask: np.ndarray[np.bool]
    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponsing the sigma along of the image axes).
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :param max_footprint: Footprint used by :func:`~iterative_peak_local_max`
        during maximum dilation. This sets the minimum distance between peaks.
    :type max_footprint: np.ndarray[np.bool]
    :param frame_index: Index of frame being processed, used to make warning
        about footprint more descriptive.
    :type frame_index: int, optional
    :param int max_diff: Maximum difference allowed for contiguous points to be
        considered a plateau when looking for a stationary value of the number
        of detected nuclei with respect to the distance between peaks as set by
        the number of iterations of the footprint. Defaults to 1.
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection. Defaults to 3, only used as fallback if plateau finding fails.
    :return: Array of integer labels of the same shape as image.
    :rtype: np.ndarray
    """
    # Band-pass filter image using difference of gaussians - this seems to work
    # better than trying to do blob detection by varying sigma on an approximation of
    # a Laplacian of Gaussian filter.
    dog = difference_of_gaussians(stack, low_sigma=low_sigma, high_sigma=high_sigma)

    # Find local minima of the bandpass-filtered image to localize nuclei
    try:
        frame_index = kwargs["frame_index"]
    except KeyError:
        frame_index = None

    kwargs["frame_index"] = frame_index

    peak_mask = iterative_peak_local_max(
        dog, footprint=max_footprint, mask=mask, **kwargs
    )

    # Generate marker mask for segmentation downstream
    markers, _ = ndi.label(peak_mask)

    return markers


# noinspection PyIncorrectDocstring
def segment_frame(stack, markers, mask, *, watershed_method, **kwargs):
    """
    Segments nuclei in a z-stack using watershed method, starting from a set of
    markers.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: np.ndarray
    :param markers: Boolean array of dimensions matching movie, with nuclei containing
        (ideally) a single unique integer value, and all other values being 0. This is
        used to perform the watershed segmentation.
    :type markers: np.ndarray[np.int]
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations).
    :type mask: np.ndarray[np.bool]
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :return: A labeled array of the same type and shape as markers, with each label
        corresponding to a mask for a single nucleus, assigned to an integer value.
    :rtype: np.ndarray
    """
    # Normalize image
    stack = img_as_float32(stack)
    stack = stack / np.max(np.abs(stack))

    # Segmentation step
    if watershed_method == "raw":
        watershed_landscape = -stack

    elif watershed_method == "distance_transform":
        watershed_landscape = -(ndi.distance_transform_edt(stack))

    elif watershed_method == "sobel":
        watershed_landscape = sobel(stack)

    else:
        raise Exception("Unrecognized watershed_method parameter")

    labels = watershed(
        watershed_landscape, markers=markers, mask=mask, compactness=0.05
    )

    # Remove small objects if a min_size parameter is provided
    try:
        min_size = kwargs["min_size"]
        remove_small_objects(labels, min_size=min_size, out=labels)
    except KeyError:
        pass

    return labels


# noinspection PyIncorrectDocstring
def denoise_movie(movie, *, denoising, **kwargs):
    """
    Denoises a movie frame-by-frame using specified method (gaussian or median
    filtering).

    :param movie: 2D (projected) or 3D image of a nuclear marker.
    :type movie: np.ndarray
    :param denoising: Determines which method to use for initial denoising of the
        image (before any filtering or morphological operations) between a gaussian
        filter and a median filter.

        * ``gaussian``: requires a ``denoising_sigma`` keyword argument to determine
          the sigma parameter for the gaussian filter.
        * ``median``: requires a ``median_footprint`` keyword argument to determine
          the footprint used for the median filter.

    :type denoising: {'gaussian', 'median'}
    :type denoising_sigma: scalar or sequence of scalars, only required if using
        ``denoising='gaussian'``.
    :param median_footprint: Footprint used for median filter denoising of the image
        prior to any morphological operations or other filtering.
    :type median_footprint: np.ndarray[np.bool]
    :return: Denoised movie.
    :rtype: np.ndarray
    """
    # Store parameters for segmentation array
    movie_shape = movie.shape
    num_timepoints = movie_shape[0]

    # Loop over frames of movie
    denoised_movie = np.empty(movie_shape, dtype=np.float32)
    for i in range(num_timepoints):
        denoised_movie[i] = denoise_frame(movie[i], denoising=denoising, **kwargs)

    return denoised_movie


# noinspection PyIncorrectDocstring
def binarize_movie(
    movie,
    *,
    thresholding,
    opening_footprint,
    closing_footprint,
    cc_min_span,
    background_max_span,
    background_sigma,
    background_threshold_method,
    background_dilation_footprint,
    **kwargs
):
    """
    Binarizes a movie frame-by-frame using specified thresholding method, separating
    between foreground and background.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: np.ndarray
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.

        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
          the footprint used for the local Otsu thresholding.
    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param cc_min_span: Minimum span in each axis that the largest connected component
        in a binarized image must have to be considered possible background noise. The
        key here is that large x- and y-axis span (more than a few nuclei) indicates
        either a region of high-intensity surface noise or a low signal-to-noise image
        that causes nuclei to appear connected when binarized. This flags the image for
        further processing and possible background removal.
    :type cc_min_span: np.ndarray
    :param background_max_span: Maximum span in each axis that a connected component
        flagged as surface noise can have and still be consisdered surface noise (i.e.
        if a large part of the stack is surface noise, the data might not be usable
        in the first place and will be very difficult to segment in 3D so we stop the
        analysis). Connected blurred regions with bounding boxes spanning this parameter
        will be removed from consideration when thresholding to obtain a nuclear mask.
    :type background_max_span: np.ndarray
    :param background_sigma: Standard deviation to use in each axis for Gaussian blurring
        of image prior to segmenting out the surface noise. This should be small in z
        so as not to bleed through the z-stack, and much larger than the nuclei in x- and
        y- so as to only keep low-frequency noise.
    :type background_sigma: np.ndarray
    :param background_threshold_method: Method to use for thresholding the Gaussian-
        blurred image for surface noise detection. Only global Otsu and Li methods
        are implemented.
    :type background_threshold_method: {"otsu", "li"}
    :param background_dilation_footprint: Structuring element used for binary dilation
        of the background surface noise mask when removing background noise.
    :type background_dilation_footprint: np.ndarray[np.bool]
    :param opening_footprint: Footprint used for closing operation.
    :type opening_footprint: np.ndarray[np.bool]
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: np.ndarray[np.bool]
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: np.ndarray[np.bool]
    :return: A boolean array of the same type and shape as stack, with True values
        corresponding to the foreground and False to the background.
    :rtype: np.ndarray
    """
    # Store parameters for segmentation array
    movie_shape = movie.shape
    num_timepoints = movie_shape[0]

    # Loop over frames of movie
    binarized_movie = np.empty(movie_shape, dtype=bool)
    for i in range(num_timepoints):
        binarized_movie[i] = binarize_frame(
            movie[i],
            thresholding=thresholding,
            opening_footprint=opening_footprint,
            closing_footprint=closing_footprint,
            cc_min_span=cc_min_span,
            background_max_span=background_max_span,
            background_sigma=background_sigma,
            background_threshold_method=background_threshold_method,
            background_dilation_footprint=background_dilation_footprint,
            **kwargs,
        )

    return binarized_movie


# noinspection PyIncorrectDocstring
def mark_movie(movie, mask, *, low_sigma, high_sigma, max_footprint, **kwargs):
    """
    Uses a difference of gaussians bandpass filter to enhance nuclei, then a local
    maximum to find markers for each nucleus. Being permissive with the filtering at
    this stage is recommended, since further filtering of the nuclear localization can
    be done post-segmentation using the size and morphology of the segmented objects.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: np.ndarray
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as movie.
    :type mask: np.ndarray[np.bool]
    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponsing the sigma along of the image axes).
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :param max_footprint: Footprint used by :func:`~iterative_peak_local_max`
        during maximum dilation. This sets the minimum distance between peaks.
    :type max_footprint: np.ndarray[np.bool]
    :param int max_diff: Maximum difference allowed for contiguous points to be
        considered a plateau when looking for a stationary value of the number
        of detected nuclei with respect to the distance between peaks as set by
        the number of iterations of the footprint. Defaults to 1.
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection. Defaults to 3, only used as fallback if plateau finding fails.
    :return: tuple(`dog`, `marker_coordinates`, `markers`) where dog is the
        bandpass-filtered image, marker_coordinates is an array of the nuclear
        locations in the image indexed as per the image (this can be used for
        visualization) and markers is a boolean array of the same shape as image, with
        the marker positions given by a True value.
    :rtype: tuple
    """
    # Store parameters for segmentation array
    movie_shape = movie.shape
    num_timepoints = movie_shape[0]

    # Loop over frames of movie
    markers = np.empty(movie_shape, dtype=np.uint32)
    for i in range(num_timepoints):
        markers[i] = mark_frame(
            movie[i],
            mask[i],
            low_sigma=low_sigma,
            high_sigma=high_sigma,
            max_footprint=max_footprint,
            frame_index=i,
            **kwargs,
        )

    return markers


# noinspection PyIncorrectDocstring
def segment_movie(movie, markers, mask, *, watershed_method, **kwargs):
    """
    Segments nuclei in a movie using watershed method.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: np.ndarray
    :param markers: Boolean array of dimensions matching movie, with nuclei containing
        (ideally) a single unique integer value, and all other values being 0. This is
        used to perform the watershed segmentation.
    :type markers: np.ndarray[np.int]
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as movie.
    :type mask: np.ndarray[np.bool]
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie.
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :return: Array with each label corresponding to a mask for a single
        nucleus, assigned to an integer value (both of the same shape as movie).
    :rtype: np.ndarray[np.int]
    """
    # Store parameters for segmentation array
    movie_shape = movie.shape
    num_timepoints = movie_shape[0]

    # Loop over frames of movie
    labels = np.empty(movie_shape, dtype=np.uint32)
    for i in range(num_timepoints):
        labels[i] = segment_frame(
            movie[i], markers[i], mask[i], watershed_method=watershed_method, **kwargs
        )

    return labels


# noinspection PyIncorrectDocstring
def denoise_movie_parallel(movie, *, denoising, client, **kwargs):
    """
    Denoises a movie frame-by-frame using specified method (gaussian or median
    filtering). This is parallelized across a Dask LocalCluster.

    :param movie: 2D (projected) or 3D image of a nuclear marker.
    :type movie: {np.ndarray, list}
    :param denoising: Determines which method to use for initial denoising of the
        image (before any filtering or morphological operations) between a gaussian
        filter and a median filter.

        * ``gaussian``: requires a ``denoising_sigma`` keyword argument to determine
          the sigma parameter for the gaussian filter.
        * ``median``: requires a ``median_footprint`` keyword argument to determine
          the footprint used for the median filter.

    :type denoising: {'gaussian', 'median'}
    :type denoising_sigma: scalar or sequence of scalars, only required if using
        ``denoising='gaussian'``.
    :param median_footprint: Footprint used for median filter denoising of the image
        prior to any morphological operations or other filtering.
    :type median_footprint: np.ndarray[np.bool]
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: tuple(`denoised_movie`, `denoised_movie_futures`, `scattered_movie`)
        where

        * `denoised_movie` is the fully evaluated denoised movie as an ndarray of the
          same shape and `dtype` as `movie`.
        * `denoised_movie_futures` is the list of futures objects resulting from the
          denoising in the worker memories before gathering and concatenation.
        * `scattered_movie` is a list of futures pointing to the input movie in
          the workers' memory, wrapped in a list.

    :rtype: tuple

    .. note::

        This function can also pass along any kwargs taken by
        :func:`~utils.parallel_computing.parallelize`.

    """
    denoise_movie_func = partial(
        denoise_movie,
        denoising=denoising,
        **kwargs,
    )

    if client is not None:
        evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
            kwargs
        )

        (
            denoised_movie,
            denoised_movie_futures,
            scattered_movie,
        ) = parallel_computing.parallelize(
            [movie],
            denoise_movie_func,
            client,
            evaluate=evaluate,
            futures_in=futures_in,
            futures_out=futures_out,
        )

    else:
        denoised_movie = denoise_movie_func(movie)
        denoised_movie_futures = None
        scattered_movie = None

    return denoised_movie, denoised_movie_futures, scattered_movie


# noinspection PyIncorrectDocstring
def binarize_movie_parallel(
    movie,
    *,
    thresholding,
    opening_footprint,
    closing_footprint,
    cc_min_span,
    background_max_span,
    background_sigma,
    background_threshold_method,
    background_dilation_footprint,
    client,
    **kwargs
):
    """
    Binarizes a movie frame-by-frame using specified thresholding method, separating
    between foreground and background. This is parallelized across a Dask
    LocalCluster.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: {np.ndarray, list}
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.

        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
          the footprint used for the local Otsu thresholding.

    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param opening_footprint: Footprint used for closing operation.
    :type opening_footprint: np.ndarray[np.bool]
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: np.ndarray[np.bool]
    :param cc_min_span: Minimum span in each axis that the largest connected component
        in a binarized image must have to be considered possible background noise. The
        key here is that large x- and y-axis span (more than a few nuclei) indicates
        either a region of high-intensity surface noise or a low signal-to-noise image
        that causes nuclei to appear connected when binarized. This flags the image for
        further processing and possible background removal.
    :type cc_min_span: np.ndarray
    :param background_max_span: Maximum span in each axis that a connected component
        flagged as surface noise can have and still be consisdered surface noise (i.e.
        if a large part of the stack is surface noise, the data might not be usable
        in the first place and will be very difficult to segment in 3D so we stop the
        analysis). Connected blurred regions with bounding boxes spanning this parameter
        will be removed from consideration when thresholding to obtain a nuclear mask.
    :type background_max_span: np.ndarray
    :param background_sigma: Standard deviation to use in each axis for Gaussian blurring
        of image prior to segmenting out the surface noise. This should be small in z
        so as not to bleed through the z-stack, and much larger than the nuclei in x- and
        y- so as to only keep low-frequency noise.
    :type background_sigma: np.ndarray
    :param background_threshold_method: Method to use for thresholding the Gaussian-
        blurred image for surface noise detection. Only global Otsu and Li methods
        are implemented.
    :type background_threshold_method: {"otsu", "li"}
    :param background_dilation_footprint: Structuring element used for binary dilation
        of the background surface noise mask when removing background noise.
    :type background_dilation_footprint: np.ndarray[np.bool]
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: np.ndarray[np.bool]
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: tuple(`binarized_movie`, `binarized_movie_futures`, `scattered_movie`)
        where

        * `binarized_movie` is the fully evaluated binarized movie as an ndarray of
          booleans of the same shape as movie, with only the pixels in the foreground
          corresponding to a `True` value.

        * `binarized_movie_futures` is the list of futures objects resulting from the
          binarization in the worker memories before gathering and concatenation.

        * `scattered_movie` is a list of futures pointing to the input movie in
          the workers' memory, wrapped in a list.

    :rtype: tuple

    .. note::

        This function can also pass along any kwargs taken by
        :func:`~utils.parallel_computing.parallelize`.

    """
    binarize_movie_func = partial(
        binarize_movie,
        thresholding=thresholding,
        opening_footprint=opening_footprint,
        closing_footprint=closing_footprint,
        cc_min_span=cc_min_span,
        background_max_span=background_max_span,
        background_sigma=background_sigma,
        background_threshold_method=background_threshold_method,
        background_dilation_footprint=background_dilation_footprint,
        **kwargs,
    )

    if client is not None:
        evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
            kwargs
        )

        (
            binarized_movie,
            binarized_movie_futures,
            scattered_movie,
        ) = parallel_computing.parallelize(
            [movie],
            binarize_movie_func,
            client,
            evaluate=evaluate,
            futures_in=futures_in,
            futures_out=futures_out,
        )

    else:
        binarized_movie = binarize_movie_func(movie)
        binarized_movie_futures = None
        scattered_movie = None

    return binarized_movie, binarized_movie_futures, scattered_movie


# noinspection PyIncorrectDocstring
def mark_movie_parallel(
    movie, mask, *, low_sigma, high_sigma, max_footprint, client, **kwargs
):
    """
    Uses a difference of gaussians bandpass filter to enhance nuclei, then a local
    maximum to find markers for each nucleus. Being permissive with the filtering at
    this stage is recommended, since further filtering of the nuclear localization can
    be done post-segmentation using the size and morphology of the segmented objects.
    This is parallelized across a Dask LocalCluster.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: {np.ndarray, list}
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as `movie`.
    :type mask: {np.ndarray, list}
    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponsing the sigma along of the image axes).
    :type low_sigma: {np.float, tuple[np.float]}
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :type high_sigma: {np.float, tuple[np.float]}
    :param max_footprint: Footprint used by :func:`~iterative_peak_local_max`
        during maximum dilation. This sets the minimum distance between peaks.
    :type max_footprint: np.ndarray[np.bool]
    :param int max_diff: Maximum difference allowed for contiguous points to be
        considered a plateau when looking for a stationary value of the number
        of detected nuclei with respect to the distance between peaks as set by
        the number of iterations of the footprint. Defaults to 1.
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection. Defaults to 3, only used as fallback if plateau finding fails.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: tuple(`marked_movie`, `marked_movie_futures`, `scattered_movies`) where

        * `marked_movie` is the fully evaluated marked movie as an ndarray of
          booleans of the same shape as movie, with each nucleus containing a single
          `True` value.
        * `marked_movie_futures` is the list of futures objects resulting from the
          marking in the worker memories before gathering and concatenation.
        * `scattered_movies` is a list with each element corresponding to a list of
          futures pointing to the input movie and mask in the workers' memory
          respectively.

    :rtype: tuple

    .. note::

        This function can also pass along any kwargs taken by
        :func:`~utils.parallel_computing.parallelize`.

    """
    mark_movie_func = partial(
        mark_movie,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        max_footprint=max_footprint,
        **kwargs,
    )

    if client is not None:
        evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
            kwargs
        )

        (
            marked_movie,
            marked_movie_futures,
            scattered_movies,
        ) = parallel_computing.parallelize(
            [movie, mask],
            mark_movie_func,
            client,
            evaluate=evaluate,
            futures_in=futures_in,
            futures_out=futures_out,
        )

    else:
        marked_movie = mark_movie_func(movie, mask)
        marked_movie_futures = None
        scattered_movies = None

    return marked_movie, marked_movie_futures, scattered_movies


# noinspection PyIncorrectDocstring
def segment_movie_parallel(movie, markers, mask, *, watershed_method, client, **kwargs):
    """
    Segments nuclei in a movie using watershed method, parallelizing on a Dask
    LocalCluster.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: np.ndarray
    :param markers: Boolean array of dimensions matching movie, with nuclei containing
        (ideally) a single unique integer value, and all other values being 0. This is
        used to perform the watershed segmentation.
    :type markers: {np.ndarray, list}
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as movie.
    :type mask: {np.ndarray, list}
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: tuple(`segmented_movie`, `segmented_movie_futures`, `scattered_movies`)
        where

        * `segmented_movie` is the fully evaluated segmented movie as an ndarray of
          the same shape as movie with `dtype=np.uint32`, with unique integer labels
          corresponding to each nucleus.
        * `segmented_movie_futures` is the list of futures objects resulting from the
          segmentation in the worker memories before gathering and concatenation.
        * `scattered_movies` is a list with each element corresponding to a list of
          futures pointing to the input `movie`, `markers`, and `mask` in the workers'
          memory respectively.

    :rtype: tuple

    .. note::

        This function can also pass along any kwargs taken by
        :func:`~utils.parallel_computing.parallelize`.

    """
    segment_movie_func = partial(
        segment_movie, watershed_method=watershed_method, **kwargs
    )

    if client is not None:
        evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
            kwargs
        )

        (
            segmented_movie,
            segmented_movie_futures,
            scattered_movies,
        ) = parallel_computing.parallelize(
            [movie, markers, mask],
            segment_movie_func,
            client,
            evaluate=evaluate,
            futures_in=futures_in,
            futures_out=futures_out,
        )

    else:
        segmented_movie = segment_movie_func(movie, markers, mask)
        segmented_movie_futures = None
        scattered_movies = None

    return segmented_movie, segmented_movie_futures, scattered_movies
