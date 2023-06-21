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
from scipy import ndimage as ndi
from functools import partial
import multiprocessing as mp
import dask
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


def iterative_peak_local_max(image, footprint):
    """
    Find peaks in an image as coordinate list.

    :param image: 2D (projected) or 3D image of a nuclear marker.
    :type image: Numpy array.
    :param footprint: Footprint used during maximum dilation. This sets the minimum
        distance between peaks, with a single maximum within the net footprint of
        iterated maximum dilations. Can be given as Tuple(num_iter, footprint) where
        the footprint is used for maximum diation num_iter times, or as a Numpy array
        of booleans that gets used as a footprint for a single maximum dilation.
    :type footprint: {Tuple(int, ndarray), ndarray}
    :return: Coordinates of the local maxima.
    :rtype: Numpy array.
    """
    # Check footprint type to decide number of iterations
    if type(footprint) is tuple:
        num_iter = footprint[0]
        footprint = footprint[1]
    elif isinstance(footprint, np.ndarray):
        num_iter = 1
    else:
        raise TypeError("footprint parameter must being either a tuple or numpy array.")

    # We apply a maximum dilation to the image, then compare to the original image
    # such that the only points that are selected correspond to maxima within the net
    # footprint of the dilation after the iterated application of the max filter.
    image_max = np.copy(image)
    for i in range(num_iter):
        image_max = ndi.maximum_filter(image_max, footprint=footprint)

    peak_mask = image == image_max
    coords = np.transpose(np.nonzero(peak_mask))

    return coords


def mark_nuclei(stack, *, low_sigma, high_sigma, max_footprint):
    """
    Uses a difference of gaussians bandpass filter to enhance nuclei, then a local
    maximum to find markers for each nucleus. Being permissive with the filtering at
    this stage is recommended, since further filtering of the nuclear localization can
    be done post-segmentation using the size and morphology of the segmented objects.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: Numpy array.
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
    :return: Tuple(dog, marker_coordinates, markers) where dog is the
        bandpass-filtered image, marker_coordinates is an array of the nuclear
        locations in the image indexed as per the image (this can be used for
        visualization) and markers is a boolean array of the same shape as image, with
        the marker positions given by a True value.
    :rtype: Tuple of numpy arrays.
    """
    # Band-pass filter image using difference of gaussians - this seems to work
    # better than trying to do blob detection by varying sigma on an approximation of
    # a Laplacian of Gaussian filter.
    dog = difference_of_gaussians(stack, low_sigma=low_sigma, high_sigma=high_sigma)

    # Find local minima of the bandpass-filtered image to localize nuclei
    marker_coordinates = iterative_peak_local_max(
        dog,
        footprint=max_footprint,
    )

    # Generate marker mask for segmentation downstream
    mask = np.zeros(dog.shape, dtype=bool)
    mask[tuple(marker_coordinates.T)] = True
    markers, _ = ndi.label(mask)

    return (dog, marker_coordinates, markers)


def segment_nuclei_stack(
    stack,
    markers,
    *,
    denoising,
    thresholding,
    closing_footprint,
    watershed_method,
    **kwargs
):
    """
    Segments nuclei in a z-stack using watershed method, starting from a set of
    markers.

    :param stack: 2D (projected) or 3D image of a nuclear marker.
    :type stack: Numpy array.
    :param markers: Boolean array of dimensions matching stack, with nuclei containing
        (ideally) a single 'True' value, and all other values being false. This is
        used to see the watershed segmentation.
    :type markers: Numpy array of booleans.
    :param denoising: Determines which method to use for initial denoising of the
        image (before any filtering or morphological operations) between a gaussian
        filter and a median filter.
        * ``gaussian``: requires a ``denoising_sigma`` keyword argument to determine
        the sigma parameter for the gaussian filter.
        * ``median``: requires a ``median_footprint`` keyword argument to determine
        the footprint used for the median filter.
    :type denoising: {'gaussian', 'median'}
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.
        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
        the footprint used for the local Otsu thresholding.
    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: Numpy array of booleans.
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param denoising_sigma: Sigma used for gaussian filter denoising of the image
        prior to any morphological operations or other filtering. If given as a scalar,
        sigma is assumed to be isotropic. Can also be given as a sequence of scalars
        matching the dimensions of the image, where each element sets the sigma in the
        corresponding image axis
    :type denoising_sigma: scalar or sequence of scalars, only required if using
        ``denoising='gaussian'``.
    :param median_footprint: Footprint used for median filter denoising of the image
        prior to any morphological operations or other filtering.
    :type median_footprint: Numpy array of booleans, only required if using
        ``denoising='median'``.
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: Numpy array of booleans, only required if using
        ``thresholding='local_otsu'``.
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :return: A labeled matrix of the same type and shape as markers, with each label
        corresponding to a mask for a single nucleus, assigned to an integer value.
    :rtype: Numpy array.
    """
    # Normalize image
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

    # Thresholding step
    if thresholding == "global_otsu":
        threshold = threshold_otsu(denoised_stack)

    elif thresholding == "local_otsu":
        try:
            otsu_footprint = kwargs["otsu_footprint"]
        except KeyError:
            raise Exception(
                "Local Otsu thresholding requires an otsu_footprint parameter."
            )
        # Convert denoised stack to uint8 for rank operation
        denoised_stack = img_as_ubyte(denoised_stack)
        threshold = rank.otsu(denoised_stack, otsu_footprint)

    elif thresholding == "li":
        threshold_guess = threshold_otsu(denoised_stack)
        threshold = threshold_li(denoised_stack, initial_guess=threshold_guess)

    else:
        raise Exception("Unrecognized thresholding parameter.")

    # Binarize stack by thresholding
    binarized_stack = denoised_stack >= threshold

    # Clean up binarized image with a closing operation
    binarized_stack = binary_closing(binarized_stack, closing_footprint)

    # Segmentation step
    if watershed_method == "raw":
        watershed_landscape = -denoised_stack

    elif watershed_method == "distance_transform":
        watershed_landscape = -(ndi.distance_transform_edt(binarized_stack))

    elif watershed_method == "sobel":
        watershed_landscape = sobel(denoised_stack)

    else:
        raise Exception("Unrecognized watershed_method parameter")

    labels = watershed(watershed_landscape, markers=markers, mask=binarized_stack)

    # Remove small objects if a min_size parameter is provided
    try:
        min_size = kwargs["min_size"]
        remove_small_objects(labels, min_size=min_size, out=labels)
    except KeyError:
        pass

    return labels


@dask.delayed
def segment_frame(
    stack,
    *,
    low_sigma,
    high_sigma,
    max_footprint,
    denoising,
    thresholding,
    closing_footprint,
    watershed_method,
    **kwargs
):
    """
    Segments nuclei in a z-stack using watershed method. Marked as delayed using
    dask decorator for subsequent parallelization over timepoints.


    :param stack: 2D (projected) or 3D z-stack of a nuclear marker.
    :type stack: Numpy array.
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
    :param denoising: Determines which method to use for initial denoising of the
        image (before any filtering or morphological operations) between a gaussian
        filter and a median filter.
        * ``gaussian``: requires a ``denoising_sigma`` keyword argument to determine
        the sigma parameter for the gaussian filter.
        * ``median``: requires a ``median_footprint`` keyword argument to determine
        the footprint used for the median filter.
    :type denoising: {'gaussian', 'median'}
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.
        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
        the footprint used for the local Otsu thresholding.
    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: Numpy array of booleans.
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie.
    :param denoising_sigma: Sigma used for gaussian filter denoising of the image
        prior to any morphological operations or other filtering. If given as a scalar,
        sigma is assumed to be isotropic. Can also be given as a sequence of scalars
        matching the dimensions of the image, where each element sets the sigma in the
        corresponding image axis
    :type denoising_sigma: scalar or sequence of scalars, only required if using
        ``denoising='gaussian'``.
    :param median_footprint: Footprint used for median filter denoising of the image
        prior to any morphological operations or other filtering.
    :type median_footprint: Numpy array of booleans, only required if using
        ``denoising='median'``.
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: Numpy array of booleans, only required if using
        ``thresholding='local_otsu'``.
    :param min_size: Smallest allowable object size.
    :type min_size: int, optional
    :return: Tuple(markers, labels) where markers is a boolean array  with the
        marker positions used for the watershed transform given by a True value and
        labels is an array with each label corresponding to a mask for a single
        nucleus, assigned to an integer value (both of the same shape as movie).
    :rtype: Tuple of Numpy arrays.
    """
    _, _, markers = mark_nuclei(
        stack,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        max_footprint=max_footprint,
    )
    labels = segment_nuclei_stack(
        stack,
        markers,
        denoising=denoising,
        thresholding=thresholding,
        closing_footprint=closing_footprint,
        watershed_method=watershed_method,
        **kwargs
    )

    return markers, labels


def segment_nuclei(
    movie,
    *,
    low_sigma,
    high_sigma,
    max_footprint,
    denoising,
    thresholding,
    closing_footprint,
    watershed_method,
    num_processes=1,
    **kwargs
):
    """
    Segments nuclei in a movie using watershed method.

    :param movie: 2D (projected) or 3D movie of a nuclear marker.
    :type stack: Numpy array.
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
    :param denoising: Determines which method to use for initial denoising of the
        image (before any filtering or morphological operations) between a gaussian
        filter and a median filter.
        * ``gaussian``: requires a ``denoising_sigma`` keyword argument to determine
        the sigma parameter for the gaussian filter.
        * ``median``: requires a ``median_footprint`` keyword argument to determine
        the footprint used for the median filter.
    :type denoising: {'gaussian', 'median'}
    :param thresholding: Determines which method to use to determine a threshold
        for binarizing the stack, between global and local Otsu threholding, and
        Li's cross-entropy minimization method.
        * ``local_otsu``: requires a ``otsu_footprint`` keyword argument to determine
        the footprint used for the local Otsu thresholding.
    :type thresholding: {'global_otsu', 'local_otsu', 'li'}
    :param closing_footprint: Footprint used for closing operation.
    :type closing_footprint: Numpy array of booleans.
    :param watershed_method: Determines what to use as basins for the watershed
        segmentation, between the inverted denoised image itself (works well for
        bright nuclear markers), the distance-transformed binarized image, and the
        sobel gradient of the image.
    :type watershed_method: {'raw', 'distance_transform', 'sobel'}
    :param int num_processes: Number of worker processes used in parallel loop over
        frames of movie.
    :param denoising_sigma: Sigma used for gaussian filter denoising of the image
        prior to any morphological operations or other filtering. If given as a scalar,
        sigma is assumed to be isotropic. Can also be given as a sequence of scalars
        matching the dimensions of the image, where each element sets the sigma in the
        corresponding image axis
    :type denoising_sigma: scalar or sequence of scalars, only required if using
        ``denoising='gaussian'``.
    :param median_footprint: Footprint used for median filter denoising of the image
        prior to any morphological operations or other filtering.
    :type median_footprint: Numpy array of booleans, only required if using
        ``denoising='median'``.
    :param otsu_footprint: Footprint used for local (rank) Otsu thresholding of the
        image for binarization.
    :type otsu_thresholding: Numpy array of booleans, only required if using
        ``thresholding='local_otsu'``.
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
    movie_dtype = movie.dtype
    num_timepoints = movie_shape[0]

    # Create partial function to run through map in parallel processes
    segment_frame_func = partial(
        segment_frame,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        max_footprint=max_footprint,
        denoising=denoising,
        thresholding=thresholding,
        closing_footprint=closing_footprint,
        watershed_method=watershed_method,
        **kwargs
    )

    # Loop over frames of movie
    movie_frames = np.split(movie, num_timepoints)
    movie_frames = [frame[0] for frame in movie_frames]
    segmentation = map(segment_frame_func, movie_frames)

    # Set up dask client for computation
    with LocalCluster(
        n_workers=int(min(0.9 * mp.cpu_count(), num_processes)),
        processes=True,
        threads_per_worker=1,
    ) as cluster, Client(cluster) as client:
        segmentation = dask.compute(*segmentation)

    markers = [frame[0] for frame in segmentation]
    labels = [frame[1] for frame in segmentation]

    # Reconstruct arrays from lists
    markers = np.stack(markers)
    labels = np.stack(labels)

    return markers, labels