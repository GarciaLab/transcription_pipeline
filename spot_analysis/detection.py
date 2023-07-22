from skimage.filters import difference_of_gaussians, threshold_triangle
from skimage.measure import label
from skimage.morphology import remove_small_objects
import numpy as np
from functools import partial
from utils import parallel_computing
from utils.image_histogram import histogram
from tracking import track_features
from utils import neighborhood_manipulation


def _make_histogram(movie, hist_range, nbins=256):
    """
    Constructs histogram of pixel values for a movie for automatic thresholding. This
    is used as a utility function to split up the code in
    `skimage.filters.threshold_triangle` to allow better parallelization since
    constructing the histogram is one of the bigger bottlenecks when automatically
    thresholding large movie arrays.
    """
    # Taken from `skimage.filters.threshold_triangle`.

    # nbins is ignored for integer arrays
    # so, we recalculate the effective nbins.
    hist, bin_centers = histogram(movie.reshape(-1), nbins, hist_range, normalize=False)
    nbins = len(hist)

    return hist, bin_centers


def _threshold_triangle(hist, bin_centers, nbins):
    """
    Finds an automatic threshold from a histogram using the triangle algorithm. This
    is used as a utility functiont to split up the code in
    `skimage.filters.threshold_triangle` to allow better parallelization since
    constructing the histogram is one of the bigger bottlenecks when automatically
    thresholding large movie arrays, but computing the theshold from a histogram
    is quite fast and should be done on a histogram for the whole movie.
    """
    # Taken from `skimage.filters.threshold_triangle`.

    # Find peak, lowest and highest gray levels.
    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.flatnonzero(hist)[[0, -1]]

    # Flip is True if left tail is shorter.
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    # If flip == True, arg_high_level becomes incorrect
    # but we don't need it anymore.
    del arg_high_level

    # Set up the coordinate system.
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]

    # Normalize.
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    # Maximize the length.
    # The ImageJ implementation includes an additional constant when calculating
    # the length, but here we omit it as it does not affect the location of the
    # minimum.
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]


def _bandpass_movie(movie, low_sigma, high_sigma):
    """
    Runs bandpass filter using `skimage.filters.difference_of_gaussians` on each
    frame of a movie before collating the resuls in a bandpass-filtered movie of the
    same shape as input `movie`.
    """
    bandpassed_movie = np.empty_like(movie, dtype=float)

    num_timepoints = movie.shape[0]
    for i in range(num_timepoints):
        bandpassed_movie[i] = difference_of_gaussians(movie[i], low_sigma, high_sigma)

    return bandpassed_movie


def _make_spot_labels(movie, threshold, min_size, connectivity):
    """
    Thresholds `movie`, labels connected components in binarized mask as per specified
    connectivity using `skimage.measure.label` and removes objects below size `min_size`.
    """
    try:
        spot_labels = np.empty(movie.shape, dtype=np.uint32)
        num_timepoints = spot_labels.shape[0]
        for i in range(num_timepoints):
            remove_small_objects(
                label(movie[i] > threshold, connectivity=connectivity),
                min_size=min_size,
                out=spot_labels[i],
            )
    except TypeError:
        raise Exception("`threshold` option not supported.")

    return spot_labels


def _bandpass_movie_parallel(movie, low_sigma, high_sigma, client, **kwargs):
    """
    Runs `skimage.filters.difference_of_gaussians` parallelized across a Dask
    Client passed as `client` on a movie with time along the 0-th axis.
    """
    dog_func = partial(_bandpass_movie, low_sigma=low_sigma, high_sigma=high_sigma)

    evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
        kwargs
    )

    bandpassed_movie, bandpassed_movie_futures, _ = parallel_computing.parallelize(
        [movie],
        dog_func,
        client,
        evaluate=evaluate,
        futures_in=futures_in,
        futures_out=futures_out,
    )
    return bandpassed_movie, bandpassed_movie_futures


def _make_spot_labels_parallel(
    movie, threshold, min_size, connectivity, client, **kwargs
):
    """
    Thresholds `movie`, labels connected components in binarized mask as per specified
    connectivity using `skimage.measure.label` and removes objects below size `min_size`
    parallelized across a Dask Dlient passed as `client`.
    """
    spot_labels_func = partial(
        _make_spot_labels,
        threshold=threshold,
        min_size=min_size,
        connectivity=connectivity,
    )

    evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
        kwargs
    )

    spot_labels, _, _ = parallel_computing.parallelize(
        [movie],
        spot_labels_func,
        client,
        evaluate=evaluate,
        futures_in=futures_in,
        futures_out=futures_out,
    )

    return spot_labels


def make_spot_mask(
    spot_movie,
    *,
    low_sigma,
    high_sigma,
    nbins=256,
    threshold="triangle",
    min_size=0,
    connectivity=None,
    return_bandpass=False,
    client=None,
):
    """
    Constructs a labelled mask separating spots from background, bandpassing and
    thresholding the image and removing objects smaller than the specified size.
    If a Dask Client is passed as a `client` kwarg, the bandpass filtering and
    thresholding will be parallelized across the client.

    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponsing the sigma along of the image axes).
    :type low_sigma: scalar or tuple of scalars
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :type high_sigma: scalar or tuple of scalars
    :param int nbins: Number of bins used to construct image histogram for automatic
        thresholding.
    :param threshold: Threshold below which to clip `spot_movie` after bandpass filter.
        Note that bandpass filtering forces a conversion to normalized float, so the
        threshold should not exceed 1. Setting `threshold="triangle"` uses automatic
        thresholding using the triangle method.
    :type threshold: {"triangle", float}
    :param int min_size: The smallest allowable object size.
    :param int connectivity: The connectivity defining the neighborhood of a pixel
        during small object removal.
    :param bool return_bandpass: If True, returns bandpass-filtered movie as
        second element of output. Otherwise returns none as second output.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: Labelled mask of spots in `spot_movie`.
    :rtype: Numpy array of integers.
    """
    if client is None:
        bandpassed_movie = _bandpass_movie(spot_movie, low_sigma, high_sigma)

        if threshold == "triangle":
            spot_threshold = threshold_triangle(bandpassed_movie)
        else:
            spot_threshold = threshold

        spot_labels = _make_spot_labels(
            bandpassed_movie, spot_threshold, min_size, connectivity
        )
    else:
        bandpassed_movie, bandpassed_movie_futures = _bandpass_movie_parallel(
            spot_movie,
            low_sigma,
            high_sigma,
            client,
            evaluate=return_bandpass,
            futures_in=False,
            futures_out=True,
        )

        if threshold == "triangle":
            def range_func(x):
                return [np.min(x), np.max(x)]

            chunked_histogram_range = client.map(range_func, bandpassed_movie_futures)
            histogram_range_array = np.array(client.gather(chunked_histogram_range))
            histogram_range = (
                histogram_range_array[:, 0].min(),
                histogram_range_array[:, 1].max(),
            )

            num_processes = len(client.scheduler_info()["workers"])
            histogram_range_scatter = client.scatter([histogram_range] * num_processes)

            histogram_func = partial(_make_histogram, nbins=nbins)

            chunked_histograms = client.map(
                histogram_func, bandpassed_movie_futures, histogram_range_scatter
            )
            gather_histograms = client.gather(chunked_histograms)

            histograms_array = np.array(gather_histograms)
            global_histogram = histograms_array[:, 0].sum(axis=0)
            global_bin_centers = histograms_array[:, 1][0]
            nbins = len(global_histogram)

            spot_threshold = _threshold_triangle(
                global_histogram, global_bin_centers, nbins
            )

        else:
            spot_threshold = threshold

        spot_labels = _make_spot_labels_parallel(
            bandpassed_movie_futures,
            spot_threshold,
            min_size,
            connectivity,
            client,
            evaluate=True,
            futures_in=False,
            futures_out=False,
        )

    if not return_bandpass:
        bandpassed_movie = None

    return spot_labels, bandpassed_movie


def detect_spots(
    spot_movie,
    *,
    low_sigma,
    high_sigma,
    frame_metadata,
    nbins=256,
    threshold="triangle",
    min_size=0,
    connectivity=None,
    return_bandpass=False,
    return_spot_mask=False,
    extra_properties=tuple(),
    drop_reverse_time=True,
    client=None,
):
    """
    Constructs a trackpy-compatible pandas Dataframe of proposed spot locations after
    bandpass-filtering, thresholding and removing small objects. If a Dask Client is
    passed as a `client` kwarg, the bandpass filtering and thresholding will be
    parallelized across the client.

    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponsing the sigma along of the image axes).
    :type low_sigma: scalar or tuple of scalars
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :type high_sigma: scalar or tuple of scalars
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :param int nbins: Number of bins used to construct image histogram for automatic
        thresholding.
    :param threshold: Threshold below which to clip `spot_movie` after bandpass filter.
        Note that bandpass filtering forces a conversion to normalized float, so the
        threshold should not exceed 1. Setting `threshold="triangle"` uses automatic
        thresholding using the triangle method.
    :type threshold: {"triangle", float}
    :param int min_size: The smallest allowable object size.
    :param int connectivity: The connectivity defining the neighborhood of a pixel
        during small object removal.
    :param bool return_bandpass: If True, returns bandpass-filtered movie as
        third element of output. Otherwise returns none as third output.
    :param bool return_spot_mask: If True, returns labelled spot mask as
        second element of output. Otherwise returns none as second output.
    :param extra_properties: Properties of each labelled region in the segmentation
        mask to measure and add to the DataFrame. With no extra properties, the
        DataFrame will have columns only for the frame, label, and centroid
        coordinates.
    :type extra_properties: Tuple of strings, optional.
    :param bool drop_reverse_time: If True, drops the columns with reversed frame
        numbers added to facilitate tracking (if you are using nuclear tracking to
        track your spots instead of tracking the spots independently, you might not
        need these columns).
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: trackpy-compatible pandas DataFrame of spot positions and extra requested
        properties.
    :rtype: pandas DataFrame
    """
    spot_mask, bandpassed_movie = make_spot_mask(
        spot_movie,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        nbins=nbins,
        threshold=threshold,
        min_size=min_size,
        connectivity=connectivity,
        return_bandpass=return_bandpass,
        client=client,
    )

    spot_dataframe = track_features.segmentation_df(
        spot_mask, spot_movie, frame_metadata, extra_properties=extra_properties
    )

    if drop_reverse_time:
        spot_dataframe.drop(
            labels=["frame_reverse", "t_frame_reverse"], axis=1, inplace=True
        )

    if not return_spot_mask:
        spot_mask = None

    return spot_dataframe, spot_mask, bandpassed_movie


def _add_neighborhood_row(
    spot_dataframe_row,
    movie,
    span,
    pos_columns=["z", "y", "x"],
):
    """
    Extracts neighborhood specified by row in dataframe generated by `detect_spots`.
    """
    t = spot_dataframe_row["frame"] - 1
    spatial_coordinates = spot_dataframe_row[pos_columns]
    coordinates = np.array([t, *spatial_coordinates])
    # Extracting single-frame neighborhood requires specifying a span of 1 in the
    # time-coordinate
    span = np.array([1, *span])

    neighborhood, coordinates_start = neighborhood_manipulation.extract_neighborhood(
        movie, coordinates, span
    )

    if neighborhood is not None:
        neighborhood = neighborhood[0]

    return neighborhood, coordinates_start


def _add_neighborhoods_to_dataframe(
    movie,
    span,
    spot_dataframe,
    pos_columns=["z", "y", "x"],
):
    """
    Extracts neighborhood specified by each row in dataframe generated by `detect_spots`,
    adding the neighborhood in-place to the dataframe in a "raw_spot" column.
    """
    spot_dataframe["raw_spot"] = None
    spot_dataframe["raw_spot"] = spot_dataframe["raw_spot"].astype(object)

    spot_dataframe["coordinates_start"] = None
    spot_dataframe["coordinates_start"] = spot_dataframe["coordinates_start"].astype(
        object
    )

    spot_dataframe[["raw_spot", "coordinates_start"]] = spot_dataframe.apply(
        _add_neighborhood_row,
        args=(movie, span, pos_columns),
        axis="columns",
        result_type="expand",
    )

    return None


def detect_and_gather_spots(
    spot_movie,
    *,
    low_sigma,
    high_sigma,
    frame_metadata,
    span,
    pos_columns=["z", "y", "x"],
    nbins=256,
    threshold="triangle",
    min_size=0,
    connectivity=None,
    return_bandpass=False,
    return_spot_mask=False,
    extra_properties=tuple(),
    drop_reverse_time=True,
    client=None,
):
    """
    Constructs a trackpy-compatible pandas Dataframe of proposed spot locations after
    bandpass-filtering, thresholding and removing small objects, adding a column for
    the raw spot data. If a Dask Client is passed as a `client` kwarg, the bandpass
    filtering and thresholding will be parallelized across the client.

    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponsing the sigma along of the image axes).
    :type low_sigma: scalar or tuple of scalars
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :type high_sigma: scalar or tuple of scalars
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :param span: Size of neighborhood to extract (rounded in each axis to the largest
        odd number below `span` if even).
    :type span: Array-like.
    :param pos_columns: Name of columns in DataFrame measurement table obtained from
        spot label array containing a position coordinate, in order of indexing of
        the input `spot_movie`.
    :type pos_columns: list of DataFrame column names
    :param int nbins: Number of bins used to construct image histogram for automatic
        thresholding.
    :param threshold: Threshold below which to clip `spot_movie` after bandpass filter.
        Note that bandpass filtering forces a conversion to normalized float, so the
        threshold should not exceed 1. Setting `threshold="triangle"` uses automatic
        thresholding using the triangle method.
    :type threshold: {"triangle", float}
    :param int min_size: The smallest allowable object size.
    :param int connectivity: The connectivity defining the neighborhood of a pixel
        during small object removal.
    :param bool return_bandpass: If True, returns bandpass-filtered movie as
        third element of output. Otherwise returns none as third output.
    :param bool return_spot_mask: If True, returns labelled spot mask as
        second element of output. Otherwise returns none as second output.
    :param extra_properties: Properties of each labelled region in the segmentation
        mask to measure and add to the DataFrame. With no extra properties, the
        DataFrame will have columns only for the frame, label, and centroid
        coordinates.
    :type extra_properties: Tuple of strings, optional.
    :param bool drop_reverse_time: If True, drops the columns with reversed frame
        numbers added to facilitate tracking (if you are using nuclear tracking to
        track your spots instead of tracking the spots independently, you might not
        need these columns).
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: trackpy-compatible pandas DataFrame of spot positions and extra requested
        properties.
    :rtype: pandas DataFrame
    """
    spot_dataframe, spot_mask, bandpassed_movie = detect_spots(
        spot_movie,
        frame_metadata=frame_metadata,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        threshold=threshold,
        min_size=min_size,
        connectivity=connectivity,
        return_bandpass=return_bandpass,
        return_spot_mask=return_spot_mask,
        drop_reverse_time=drop_reverse_time,
        client=client,
    )

    _add_neighborhoods_to_dataframe(spot_movie, span, spot_dataframe, pos_columns)

    return spot_dataframe, spot_mask, bandpassed_movie