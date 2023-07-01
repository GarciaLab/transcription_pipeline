import warnings
import numpy as np
from skimage.filters import difference_of_gaussians
from skimage.filters import rank
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li
from skimage.filters import sobel
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import binary_closing
from skimage.morphology import remove_small_objects
from skimage.util import img_as_ubyte
from skimage.util import img_as_float32
from scipy import ndimage as ndi
from functools import partial
import multiprocessing as mp
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster


def ellipsoid(diameter, height):
    """
    Constructs an ellipsoid footprint for morphological operations - this is usually
    better than built-in skimage.morphology footprints because the voxel dimensions
    in our images are typically anisotropic.

    :param int diameter: Diameter in xy-plane of ellipsoid footprint.
    :param int heigh: Height in z-axis of ellipsoid footprint.
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

    round_odd = lambda x: int(((x + 1) // 2) * 2 - 1)
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


def iterative_peak_local_max(image, footprint, *, mask, frame_index, **kwargs):
    """
    Find peaks in an image as coordinate list.

    :param image: 2D (projected) or 3D image of a nuclear marker (usually used
        on pre-processed images, for instance with a DoG filter).
    :type image: Numpy array.
    :param footprint: Footprint used during maximum dilation. This sets the minimum
        distance between peaks, with a single maximum within the net footprint of
        iterated maximum dilations. Can be given as a Numpy array of booleans that
        gets used as a footprint for a single maximum dilation, as a
        Tuple(num_iter, footprint) where the footprint is used for maximum diation
        num_iter times, as Tuple((num_iter_lower, num_iter_upper), footprint)
        where num_iter_lower and num_iter_upper are the lowest and highest number of
        iterations respectively that the function will try when looking for a
        stationary point in the number of detected maxima for each frame so that no
        frame is over- or under-segmented.
    :type footprint: {ndarray, Tuple(int, ndarray), Tuple((int, int), ndarray)}
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations).
    :type mask: Numpy array of booleans.
    :param frame_number: Index of frame being processed, used to make warning
        about footprint more descriptive.
    :type frame_number: int
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection.
    :return: Boolean mask labeling local maxima.
    :rtype: Numpy array.
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

    # We apply a maximum dilation to the image, then compare to the original image
    # such that the only points that are selected correspond to maxima within the net
    # footprint of the dilation after the iterated application of the max filter.
    image_max = np.copy(image)
    for _ in range(num_iter):
        image_max = ndi.maximum_filter(image_max, footprint=footprint)

    # Perform iterative max dilation in specified range and record results
    max_dilations_iter = np.empty((num_iter_range,) + image.shape, dtype=image.dtype)
    max_dilations_iter[0] = image_max
    for i in range(1, num_iter_range):
        max_dilations_iter[i] = ndi.maximum_filter(
            max_dilations_iter[i - 1], footprint=footprint
        )

    # Construct boolean mask marking local maxima for each number of iterations,
    # multiplying by mask to remove spurious peaks outside foreground
    peak_mask_iter = np.empty(max_dilations_iter.shape, dtype=bool)
    for i in range(peak_mask_iter.shape[0]):
        peak_mask_iter[i] = (max_dilations_iter[i] == image) * mask

    # Count number of marked maxima for each number of iterations
    num_peaks_iter = []
    for i in range(peak_mask_iter.shape[0]):
        num_peaks = np.sum(peak_mask_iter[i])
        num_peaks_iter.append(num_peaks)

    # Compute successive differences between number of peaks for range of iterations
    diff_num_peaks_iter = [s - t for s, t in zip(num_peaks_iter, num_peaks_iter[1:])]
    diff_num_peaks_iter = np.array(diff_num_peaks_iter, dtype=float)

    # Perform rolling average
    try:
        averaging_window = kwargs["averaging_window"]
    except KeyError:
        averaging_window = 3

    diff_num_peaks_iter = ndi.uniform_filter1d(
        diff_num_peaks_iter, size=averaging_window
    )

    # Look for stationary point
    if frame_index is not None:
        frame_message = "Frame {0}: ".format(i)
        frame_message_end = (
            " Note: Frame number inaccurate if chunking and parallelizing."
        )
    else:
        frame_message = ""
        frame_message_end = ""

    stationary_point = 0
    if num_iter_range > 1:  # Checks if iterations were requested
        stationary_point = np.argmin(diff_num_peaks_iter)

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
        if stationary_point == (diff_num_peaks_iter.size - 1):
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
        if diff_num_peaks_iter[stationary_point] > 1:
            warnings.warn(
                "".join(
                    [
                        frame_message,
                        "No stationary point, defaulting to mininum ",
                        "difference between iterations.",
                        frame_message_end,
                    ]
                ),
                stacklevel=2,
            )

    peak_mask = peak_mask_iter[stationary_point]

    return peak_mask


def denoise_frame(stack, denoising, **kwargs):
    """
    Binarizes frames in a z-stack using specified method, separating between
    foreground and background.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: Numpy array.
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
    :type median_footprint: Numpy array of booleans, only required if using
        ``denoising='median'``.
    :return: Denoised stack.
    :rtype: Numpy array.
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


def binarize_frame(stack, *, thresholding, closing_footprint, **kwargs):
    """
    Binarizes frames in a z-stack using specified method, separating between
    foreground and background.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: Numpy array.
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.
        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
        the footprint used for the local Otsu thresholding.
    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: Numpy array of booleans.
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: Numpy array of booleans, only required if using
        ``thresholding='local_otsu'``.
    :return: A boolean array of the same type and shape as stack, with True values
        corresponding to the foreground and False to the background.
    :rtype: Numpy array.
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

    # Clean up binarized image with a closing operation
    binarized_stack = binary_closing(binarized_stack, closing_footprint)

    return binarized_stack


def mark_frame(stack, mask, *, low_sigma, high_sigma, max_footprint, **kwargs):
    """
    Uses a difference of gaussians bandpass filter to enhance nuclei, then a local
    maximum to find markers for each nucleus. Being permissive with the filtering at
    this stage is recommended, since further filtering of the nuclear localization can
    be done post-segmentation using the size and morphology of the segmented objects.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: Numpy array.
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations).
    :type mask: Numpy array of booleans.
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
    :type max_footprint: Numpy array of booleans.
    :param frame_index: Index of frame being processed, used to make warning
        about footprint more descriptive.
    :type frame_index: int, optional
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection.
    :return: Array of integer labels of the same shape as image.
    :rtype: Numpy array.
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

    kwargs['frame_index'] = frame_index

    peak_mask = iterative_peak_local_max(
        dog, footprint=max_footprint, mask=mask, **kwargs
    )

    # Generate marker mask for segmentation downstream
    markers, _ = ndi.label(peak_mask)

    return markers


def segment_frame(stack, markers, mask, *, watershed_method, **kwargs):
    """
    Segments nuclei in a z-stack using watershed method, starting from a set of
    markers.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: Numpy array.
    :param markers: Boolean array of dimensions matching movie, with nuclei containing
        (ideally) a single unique integer value, and all other values being 0. This is
        used to perform the watershed segmentation.
    :type markers: Numpy array of integers.
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations).
    :type mask: Numpy array of booleans.
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :return: A labeled matrix of the same type and shape as markers, with each label
        corresponding to a mask for a single nucleus, assigned to an integer value.
    :rtype: Numpy array.
    """
    # Normalize image
    stack = img_as_float32(stack)
    stack = stack / np.max(np.abs(stack))

    # Segmentation step
    if watershed_method == "raw":
        watershed_landscape = -stack

    elif watershed_method == "distance_transform":
        watershed_landscape = -(ndi.distance_transform_edt(binarized_stack))

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


def denoise_movie(movie, *, denoising, **kwargs):
    """
    Binarizes frames in a z-stack using specified method, separating between
    foreground and background.

    :param movie: 2D (projected) or 3D image of a nuclear marker.
    :type movie: Numpy array.
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
    :type median_footprint: Numpy array of booleans, only required if using
        ``denoising='median'``.
    :return: Denoised movie.
    :rtype: Numpy array.
    """
    # Store parameters for segmentation array
    movie_shape = movie.shape
    num_timepoints = movie_shape[0]

    # Loop over frames of movie
    denoised_movie = np.empty(movie_shape, dtype=np.float32)
    for i in range(num_timepoints):
        denoised_movie[i] = denoise_frame(movie[i], denoising=denoising, **kwargs)

    return denoised_movie


def binarize_movie(movie, *, thresholding, closing_footprint, **kwargs):
    """
    Binarizes frames in a z-stack using specified method, separating between
    foreground and background.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type stack: Numpy array.
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.
        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
        the footprint used for the local Otsu thresholding.
    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: Numpy array of booleans.
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: Numpy array of booleans, only required if using
        ``thresholding='local_otsu'``.
    :return: A boolean array of the same type and shape as stack, with True values
        corresponding to the foreground and False to the background.
    :rtype: Numpy array.
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
            closing_footprint=closing_footprint,
            **kwargs,
        )

    return binarized_movie


def mark_movie(movie, mask, *, low_sigma, high_sigma, max_footprint, **kwargs):
    """
    Uses a difference of gaussians bandpass filter to enhance nuclei, then a local
    maximum to find markers for each nucleus. Being permissive with the filtering at
    this stage is recommended, since further filtering of the nuclear localization can
    be done post-segmentation using the size and morphology of the segmented objects.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: Numpy array.
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as movie.
    :type mask: Numpy array of booleans.
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
    :type max_footprint: Numpy array of booleans.
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection.
    :return: Tuple(dog, marker_coordinates, markers) where dog is the
        bandpass-filtered image, marker_coordinates is an array of the nuclear
        locations in the image indexed as per the image (this can be used for
        visualization) and markers is a boolean array of the same shape as image, with
        the marker positions given by a True value.
    :rtype: Tuple of numpy arrays.
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


def segment_movie(movie, markers, mask, *, watershed_method, **kwargs):
    """
    Segments nuclei in a movie using watershed method.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: Numpy array.
    :param markers: Boolean array of dimensions matching movie, with nuclei containing
        (ideally) a single unique integer value, and all other values being 0. This is
        used to perform the watershed segmentation.
    :type markers: Numpy array of integers.
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as movie.
    :type mask: Numpy array of booleans.
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie.
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :return: Tuple(markers, labels) where markers is a boolean array  with the
        marker positions used for the watershed transform given by a True value and
        labels is an array with each label corresponding to a mask for a single
        nucleus, assigned to an integer value (both of the same shape as movie).
    :rtype: Tuple of Numpy arrays.
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


def _convert_to_dask(movie, chunk_shape):
    """
    Converts supported array types to dask with requested chunking.
    """
    if isinstance(movie, np.ndarray):
        dask_movie = da.from_array(movie, chunks=chunk_shape)
    elif isinstance(movie, zarr.core.Array):
        dask_movie = da.from_zarr(movie, chunks=chunk_shape)
    elif isinstance(movie, da.Array):
        dask_movie = da.rechunk(movie, chunks=chunk_shape)
    else:
        raise Exception("Movie data type not recognized, must be numpy, zarr or dask.")

    return dask_movie


def _parallelization_kwargs(kwargs):
    """
    Constructs Dask Client kwarg dict for connecting to existing LocalCluster or
    creating new LocalCluster.
    """
    existing_client_kwargs = {}
    new_client_kwargs = {}

    try:
        existing_client_kwargs["address"] = kwargs["address"]
        existing_client_kwargs["timeout"] = "2s"
    except KeyError:
        existing_client_kwargs["address"] = None

    try:
        num_processes = kwargs["num_processes"]
    except KeyError:
        num_processes = 4

    try:
        new_client_kwargs["memory_limit"] = kwargs["memory_limit"]
    except KeyError:
        new_client_kwargs["memory_limit"] = "4GB"

    new_client_kwargs["n_workers"] = int(min(0.9 * mp.cpu_count(), num_processes))
    new_client_kwargs["processes"] = True
    new_client_kwargs["threads_per_worker"] = 1

    return existing_client_kwargs, new_client_kwargs


def _compute_map(movies_list, func, client_kwargs, dtype_out):
    """
    Runs a function func taking a number of identically-shaped dask arrays as
    positional arguments in parallel across chunks of a dask array on a Dask client
    generated as per options in the client_kwargs dict. The input dask arrays need
    to be packed in respective order as a list before passing to this function.
    """
    with Client(**client_kwargs) as client:
        num_processes = len(client.scheduler_info()["workers"])

        # Figure out how to split movie into chunks to distribute across processes
        num_timepoints_per_chunk = int(np.ceil(movies_list[0].shape[0] / num_processes))
        num_axes = len(movies_list[0].shape)
        chunk_shape = (num_timepoints_per_chunk,) + movies_list[0].shape[1:num_axes]

        dask_movies_list = []
        for movie in movies_list:
            dask_movies_list.append(_convert_to_dask(movie, chunk_shape))

        denoised_map = da.map_blocks(
            func, *dask_movies_list, meta=np.array((), dtype=dtype_out)
        )
        denoised_movie = denoised_map.compute()

    return denoised_movie


def _send_compute_to_cluster(movies_list, func, kwargs, dtype_out):
    """
    Sends func parallel computation to existing LocalCluster if available, falls
    back to creating one if not. Uses kwargs dictionary passed down from parent
    function to determine parallelization parameters.
    """
    existing_client_kwargs, new_client_kwargs = _parallelization_kwargs(kwargs)

    # Try to connect to provided address. If no address or cannot connect, spin
    # up local cluster
    if existing_client_kwargs["address"] is None:
        func_eval = _compute_map(movies_list, func, new_client_kwargs, dtype_out)
    else:
        try:
            func_eval = _compute_map(
                movies_list, func, existing_client_kwargs, dtype_out
            )
        except OSError:
            warnings.warn(
                "".join(
                    [
                        "Cluster not found at specified address, ",
                        "starting new cluster with default parameters.",
                    ]
                ),
                stacklevel=2,
            )
            func_eval = _compute_map(movies_list, func, new_client_kwargs, dtype_out)

    return func_eval


def denoise_movie_parallel(movie, *, denoising, **kwargs):
    """
    Binarizes frames in a z-stack using specified method, separating between
    foreground and background.

    :param movie: 2D (projected) or 3D image of a nuclear marker.
    :type movie: Numpy array.
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
    :type median_footprint: Numpy array of booleans, only required if using
        ``denoising='median'``.
    :param address: Check specified port (default is 8786) for existing
        LocalCluster to connect to.
    :type address: str or LocalCluster object
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie. Only required if not connecting to existing LocalCluster.
        Default is 4.
    :param str memory_limit: Memory limit of each dask worker for parallelization -
        this shouldn't be an issue when running our usual datasets on the server,
        but is useful if running on a different machine and seeing out-of-memory
        errors from Dask. Should be provided as a string in format '_GB'. Only
        required if not connecting to existing LocalCluster. Default is 4GB.
    :return: Denoised movie.
    :rtype: Numpy array.
    """

    denoise_movie_func = partial(
        denoise_movie,
        denoising=denoising,
        **kwargs,
    )

    denoised_movie = _send_compute_to_cluster(
        [movie], denoise_movie_func, kwargs, np.float32
    )

    return denoised_movie


def binarize_movie_parallel(movie, *, thresholding, closing_footprint, **kwargs):
    """
    Binarizes frames in a z-stack using specified method, separating between
    foreground and background. This is parallelized across a Dask LocalCluster.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type stack: Numpy array.
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.
        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
        the footprint used for the local Otsu thresholding.
    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: Numpy array of booleans.
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: Numpy array of booleans, only required if using
        ``thresholding='local_otsu'``.
    :param address: Check specified port (default is 8786) for existing
        LocalCluster to connect to.
    :type address: str or LocalCluster object
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie. Only required if not connecting to existing LocalCluster.
        Default is 4.
    :param str memory_limit: Memory limit of each dask worker for parallelization -
        this shouldn't be an issue when running our usual datasets on the server,
        but is useful if running on a different machine and seeing out-of-memory
        errors from Dask. Should be provided as a string in format '_GB'. Only
        required if not connecting to existing LocalCluster. Default is 4GB.
    :return: A boolean array of the same type and shape as stack, with True values
        corresponding to the foreground and False to the background.
    :rtype: Numpy array.
    """
    binarize_movie_func = partial(
        binarize_movie,
        thresholding=thresholding,
        closing_footprint=closing_footprint,
        **kwargs,
    )

    binarized_movie = _send_compute_to_cluster(
        [movie], binarize_movie_func, kwargs, bool
    )

    return binarized_movie


def mark_movie_parallel(movie, mask, *, low_sigma, high_sigma, max_footprint, **kwargs):
    """
    Uses a difference of gaussians bandpass filter to enhance nuclei, then a local
    maximum to find markers for each nucleus. Being permissive with the filtering at
    this stage is recommended, since further filtering of the nuclear localization can
    be done post-segmentation using the size and morphology of the segmented objects.
    This is parallelized across a Dask LocalCluster.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type movie: Numpy array.
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as movie.
    :type mask: Numpy array of booleans.
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
    :type max_footprint: Numpy array of booleans.
    :param int averaging_window: Size of averaging window used to perform moving
        average of number of detected peaks when selecting the 'elbow' in the
        detection.
    :param address: Check specified port (default is 8786) for existing
        LocalCluster to connect to.
    :type address: str or LocalCluster object
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie. Only required if not connecting to existing LocalCluster.
        Default is 4.
    :param str memory_limit: Memory limit of each dask worker for parallelization -
        this shouldn't be an issue when running our usual datasets on the server,
        but is useful if running on a different machine and seeing out-of-memory
        errors from Dask. Should be provided as a string in format '_GB'. Only
        required if not connecting to existing LocalCluster. Default is 4GB.
    :return: Tuple(dog, marker_coordinates, markers) where dog is the
        bandpass-filtered image, marker_coordinates is an array of the nuclear
        locations in the image indexed as per the image (this can be used for
        visualization) and markers is a boolean array of the same shape as image, with
        the marker positions given by a True value.
    :rtype: Tuple of numpy arrays.
    """
    mark_movie_func = partial(
        mark_movie,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        max_footprint=max_footprint,
        **kwargs,
    )

    markers = _send_compute_to_cluster(
        [movie, mask], mark_movie_func, kwargs, np.uint32
    )

    return markers


def segment_movie_parallel(movie, markers, mask, *, watershed_method, **kwargs):
    """
    Segments nuclei in a movie using watershed method, parallelizing on a Dask
    LocalCluster.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type stack: Numpy array.
    :param markers: Boolean array of dimensions matching movie, with nuclei containing
        (ideally) a single unique integer value, and all other values being 0. This is
        used to perform the watershed segmentation.
    :type markers: Numpy array of integers.
    :param mask: Mask labelling regions of interest, used to screen out spurious
        peaks when counting detected maxima (important for searching for a
        stationary value of the detected maxima with respect to the number of
        iterations). Same shape as movie.
    :type mask: Numpy array of booleans.
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param address: Check specified port (default is 8786) for existing
        LocalCluster to connect to.
    :type address: str or LocalCluster object
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie. Only required if not connecting to existing LocalCluster.
        Default is 4.
    :param str memory_limit: Memory limit of each dask worker for parallelization -
        this shouldn't be an issue when running our usual datasets on the server,
        but is useful if running on a different machine and seeing out-of-memory
        errors from Dask. Should be provided as a string in format '_GB'. Only
        required if not connecting to existing LocalCluster. Default is 4GB.
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :return: Tuple(markers, labels) where markers is a boolean array  with the
        marker positions used for the watershed transform given by a True value and
        labels is an array with each label corresponding to a mask for a single
        nucleus, assigned to an integer value (both of the same shape as movie).
    :rtype: Tuple of Numpy arrays.
    """
    segment_movie_func = partial(
        segment_movie, watershed_method=watershed_method, **kwargs
    )

    segmentation = _send_compute_to_cluster(
        [movie, markers, mask], segment_movie_func, kwargs, np.uint32
    )

    return segmentation