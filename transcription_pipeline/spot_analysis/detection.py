from skimage.filters import gaussian, difference_of_gaussians, threshold_triangle
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.util import img_as_float32
import numpy as np
import pandas as pd
from functools import partial
from ..utils import parallel_computing
from ..utils.image_histogram import histogram
from ..tracking import track_features
from ..utils import neighborhood_manipulation
import warnings


def _make_histogram(movie, hist_range, nbins=256):
    """
    Constructs histogram of pixel values for a movie for automatic thresholding. This
    is used as a utility function to split up the code in
    `skimage.filters.threshold_triangle` to allow better parallelization since
    constructing the histogram is one of the bigger bottlenecks when automatically
    thresholding large movie arrays.
    """
    # Taken from `skimage.filters.threshold_triangle`.
    nan_mask = np.isnan(movie)
    if nan_mask.any():
        masked_movie = movie[~nan_mask]
        warnings.warn("Movie has NaN values.")
    else:
        masked_movie = movie

    hist, bin_centers = histogram(
        masked_movie.reshape(-1), nbins, hist_range, normalize=False
    )

    return hist, bin_centers


def _threshold_triangle(hist, bin_centers, nbins):
    """
    Finds an automatic threshold from a histogram using the triangle algorithm. This
    is used as a utility function to split up the code in
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
    frame of a movie before collating the results in a bandpass-filtered movie of the
    same shape as input `movie`.
    """
    bandpassed_movie = np.empty_like(movie, dtype=np.float32)

    num_timepoints = movie.shape[0]
    for i in range(num_timepoints):
        frame = movie[i]

        if np.isnan(frame).sum():
            # This uses the normalized convolution trick from
            # https://stackoverflow.com/a/36307291
            nan_mask = np.isnan(frame)

            # Converting to float32 instead of float64 to save some memory.
            frame_cast_zero = img_as_float32(frame)
            frame_cast_zero[nan_mask] = 0

            frame_norm = np.ones_like(frame, dtype=np.float32)
            frame_norm[nan_mask] = 0

            low_sigma_gaussian = gaussian(frame_cast_zero, sigma=low_sigma)
            low_sigma_norm = gaussian(frame_norm, sigma=low_sigma)
            # We set the normalization zeros to 1 to avoid dividing by 0.
            # This has no side effects since this should only affect the
            # NaN values.
            low_sigma_norm[low_sigma_norm == 0] = 1
            low_sigma_frame = low_sigma_gaussian / low_sigma_norm

            high_sigma_gaussian = gaussian(frame_cast_zero, sigma=high_sigma)
            high_sigma_norm = gaussian(frame_norm, sigma=high_sigma)
            high_sigma_norm[high_sigma_norm == 0] = 1
            high_sigma_frame = high_sigma_gaussian / high_sigma_norm

            bandpassed_frame = low_sigma_frame - high_sigma_frame
            bandpassed_frame[nan_mask] = np.nan
            bandpassed_movie[i] = bandpassed_frame

        else:
            bandpassed_movie[i] = difference_of_gaussians(frame, low_sigma, high_sigma)

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

    (
        bandpassed_movie,
        bandpassed_movie_futures,
        movie_futures,
    ) = parallel_computing.parallelize(
        [movie],
        dog_func,
        client,
        evaluate=evaluate,
        futures_in=futures_in,
        futures_out=futures_out,
    )
    return bandpassed_movie, bandpassed_movie_futures, movie_futures[0]


def _make_spot_labels_parallel(
    movie, threshold, min_size, connectivity, client, **kwargs
):
    """
    Thresholds `movie`, labels connected components in binarized mask as per specified
    connectivity using `skimage.measure.label` and removes objects below size `min_size`
    parallelized across a Dask Client passed as `client`.
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

    spot_labels, spot_labels_futures, _ = parallel_computing.parallelize(
        [movie],
        spot_labels_func,
        client,
        evaluate=evaluate,
        futures_in=futures_in,
        futures_out=futures_out,
    )

    return spot_labels, spot_labels_futures


def make_spot_mask(
    spot_movie,
    *,
    low_sigma,
    high_sigma,
    nbins=256,
    threshold="triangle",
    threshold_factor=1,
    min_size=0,
    connectivity=None,
    return_bandpass=False,
    keep_futures_bandpass=True,
    return_spot_labels=True,
    keep_futures_spot_labels=True,
    keep_futures_spots_movie=True,
    client=None,
):
    """
    Constructs a labelled mask separating spots from background, bandpass filtering and
    thresholding the image and removing objects smaller than the specified size.
    If a Dask Client is passed as a `client` kwarg, the bandpass filtering and
    thresholding will be parallelized across the client.

    :param spot_movie: Movie of spot channel.
    :type spot_movie: np.ndarray
    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponsing the sigma along of the image axes).
    :type low_sigma: {float, tuple[float]}
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :type high_sigma: {float, tuple[float]}
    :param int nbins: Number of bins used to construct image histogram for automatic
        thresholding.
    :param threshold: Threshold below which to clip `spot_movie` after bandpass filter.
        Note that bandpass filtering forces a conversion to normalized float, so the
        threshold should not exceed 1. Setting `threshold="triangle"` uses automatic
        thresholding using the triangle method.
    :type threshold: {"triangle", float}
    :param float threshold_factor: If using automated thresholding, this factor is multiplied
        by the proposed threshold value. This gives some degree of control over the stringency
        of thresholding while still getting a ballpark value using the automated method.
    :param int min_size: The smallest allowable object size.
    :param int connectivity: The connectivity defining the neighborhood of a pixel
        during small object removal.
    :param bool return_bandpass: If `True`, returns bandpass-filtered movie as
        third element of output. Otherwise returns none as third output.
    :param bool keep_futures_bandpass: If `True`, keeps generated bandpass-filtered movie
        as a list of `Futures` in the Dask worker memories, returning as a list in fourth output.
    :param bool return_spot_labels: If `True`, returns labelled spot mask as
        first element of output. Otherwise returns none as first output.
    :param bool keep_futures_spot_labels: If `True`, keeps generated labelled mask as a list of
        `Futures` in the Dask worker memories, returning as a list in second output.
    :param bool keep_futures_spots_movie: If `True`, keeps input `spot_movie` as a list of
        `Futures` in the Dask worker memories, returning as a list in fifth output.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: tuple(Labelled mask of spots in `spot_movie`, labelled mask as list of
        `Futures`, bandpass-filtered movie, bandpass-filtered movie as a list of `Futures`,
        input `spot_movie` as list of `Futures`)
    :rtype: tuple
    """
    if client is None:
        bandpassed_movie = _bandpass_movie(spot_movie, low_sigma, high_sigma)

        if threshold == "triangle":
            spot_threshold = threshold_triangle(bandpassed_movie) * threshold_factor
        else:
            spot_threshold = threshold

        spot_labels = _make_spot_labels(
            bandpassed_movie, spot_threshold, min_size, connectivity
        )

        spot_movie_futures = None
        bandpassed_movie_futures = None
        spot_labels_futures = None

    else:
        (
            bandpassed_movie,
            bandpassed_movie_futures,
            spot_movie_futures,
        ) = _bandpass_movie_parallel(
            spot_movie,
            low_sigma,
            high_sigma,
            client,
            evaluate=return_bandpass,
            futures_in=keep_futures_spots_movie,
            futures_out=True,
        )

        if threshold == "triangle":

            def range_func(x):
                return [np.nanmin(x), np.nanmax(x)]

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

            spot_threshold = (
                _threshold_triangle(global_histogram, global_bin_centers, nbins)
                * threshold_factor
            )

        else:
            spot_threshold = threshold

        spot_labels, spot_labels_futures = _make_spot_labels_parallel(
            bandpassed_movie_futures,
            spot_threshold,
            min_size,
            connectivity,
            client,
            evaluate=return_spot_labels,
            futures_in=False,
            futures_out=keep_futures_spot_labels,
        )

    if not keep_futures_bandpass:
        del bandpassed_movie_futures
        bandpassed_movie_futures = None

    return (
        spot_labels,
        spot_labels_futures,
        bandpassed_movie,
        bandpassed_movie_futures,
        spot_movie_futures,
    )


def detect_spots(
    spot_movie,
    *,
    low_sigma,
    high_sigma,
    frame_metadata,
    nbins=256,
    threshold="triangle",
    threshold_factor=1,
    min_size=0,
    connectivity=None,
    return_bandpass=False,
    keep_futures_bandpass=False,
    return_spot_labels=False,
    keep_futures_spot_labels=True,
    return_spot_dataframe=True,
    keep_futures_spot_dataframe=False,
    keep_futures_spots_movie=False,
    extra_properties=tuple(),
    drop_reverse_time=True,
    client=None,
):
    """
    Constructs a trackpy-compatible pandas Dataframe of proposed spot locations after
    bandpass-filtering, thresholding and removing small objects. If a Dask Client is
    passed as a `client` kwarg, the bandpass filtering and thresholding will be
    parallelized across the client.

    :param spot_movie: Movie of spot channel.
    :type spot_movie: np.ndarray
    :param low_sigma: Sigma to use as the low-pass filter (mainly filters out
        noise). Can be given as float (assumes isotropic sigma) or as sequence/array
        (each element corresponding the sigma along of the image axes).
    :type low_sigma: {np.float, tuple[np.float]}
    :param high_sigma: Sigma to use as the high-pass filter (removes structured
        background and dims down areas where nuclei are close together that might
        start to coalesce under other morphological operations). Can be given as float
        (assumes isotropic sigma) or as sequence/array (each element corresponsing the
        sigma along of the image axes).
    :type high_sigma: {np.float, tuple[np.float]}
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :param int nbins: Number of bins used to construct image histogram for automatic
        thresholding.
    :param threshold: Threshold below which to clip `spot_movie` after bandpass filter.
        Note that bandpass filtering forces a conversion to normalized float, so the
        threshold should not exceed 1. Setting `threshold="triangle"` uses automatic
        thresholding using the triangle method.
    :type threshold: {"triangle", float}
    :param float threshold_factor: If using automated thresholding, this factor is multiplied
        by the proposed threshold value. This gives some degree of control over the stringency
        of thresholding while still getting a ballpark value using the automated method.
    :param int min_size: The smallest allowable object size.
    :param int connectivity: The connectivity defining the neighborhood of a pixel
        during small object removal.
    :param bool return_bandpass: If True, returns bandpass-filtered movie as
        third element of output. Otherwise returns none as third output.
    :param bool keep_futures_bandpass: If `True`, keeps generated bandpass-filtered movie
        as a list of `Futures` in the Dask worker memories, returning as a list in fourth output.
    :param bool return_spot_labels: If True, returns labelled spot mask as
        second element of output. Otherwise returns none as second output.
    :param bool keep_futures_spot_labels: If `True`, keeps generated labelled mask as a list of
        `Futures` in the Dask worker memories, returning as a list in second output.
    :param bool return_spot_dataframe: If True, returns fully evaluated dataframe of
        labelled spots as first output (otherwise returns `None`).
    :param bool keep_futures_spot_dataframe: If `True`, keeps dataframe of labelled
        spots as a list of `Futures` in the Dask worker memories.
    :param bool keep_futures_spots_movie: If `True`, keeps dataframe of input `spot_movie`
        as a list of `Futures` in the Dask worker memories.
    :param extra_properties: Properties of each labelled region in the segmentation
        mask to measure and add to the DataFrame. With no extra properties, the
        DataFrame will have columns only for the frame, label, and centroid
        coordinates.
    :type extra_properties: tuple[str]
    :param bool drop_reverse_time: If True, drops the columns with reversed frame
        numbers added to facilitate tracking (if you are using nuclear tracking to
        track your spots instead of tracking the spots independently, you might not
        need these columns).
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return:

        * trackpy-compatible pandas DataFrame of spot positions and extra requested
          properties.
        * trackpy-compatible pandas DataFrame of spot positions and extra requested
          properties, chunked up as list of `Futures` corresponding to labelled mask and
          bandpassed-filtered movie `Futures` (see below).
        * labelled mask of spots in `spot_movie`
        * labelled mask as list of `Futures`
        * bandpass-filtered movie
        * bandpass-filtered movie as a list of `Futures`
        * input `spot_movie` as list of `Futures`

    :rtype: tuple
    """
    if client is not None:
        evaluate_make_spot_mask = return_spot_labels
    else:
        keep_futures_spots_movie = False
        evaluate_make_spot_mask = True

    (
        spot_labels,
        spot_labels_futures,
        bandpassed_movie,
        bandpassed_movie_futures,
        spot_movie_futures,
    ) = make_spot_mask(
        spot_movie,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        nbins=nbins,
        threshold=threshold,
        threshold_factor=threshold_factor,
        min_size=min_size,
        connectivity=connectivity,
        return_spot_labels=evaluate_make_spot_mask,
        return_bandpass=return_bandpass,
        keep_futures_bandpass=keep_futures_bandpass,
        keep_futures_spot_labels=keep_futures_spot_labels,
        keep_futures_spots_movie=keep_futures_spots_movie,
        client=client,
    )

    if client is not None:

        def segmentation_df_func(
            spot_labels_chunk, spot_movie_chunk, initial_frame_index
        ):
            """
            Utility function to parallelize generation of dataframe of segmented
            spots.
            """
            spot_df, _, _ = track_features.segmentation_df(
                spot_labels_chunk,
                spot_movie_chunk,
                frame_metadata,
                initial_frame_index=initial_frame_index,
                extra_properties=extra_properties,
            )

            if drop_reverse_time:
                spot_df.drop(
                    labels=["frame_reverse", "t_frame_reverse"], axis=1, inplace=True
                )

            return spot_df

        num_frames_chunks = client.map(lambda x: x.shape[0], spot_labels_futures)
        num_frames_chunks = client.gather(num_frames_chunks)

        initial_frame_chunks = (
            np.asarray([0] + num_frames_chunks[:-1]).cumsum().tolist()
        )
        spot_dataframe_futures = client.map(
            segmentation_df_func,
            spot_labels_futures,
            spot_movie_futures,
            initial_frame_chunks,
        )

        if return_spot_dataframe:
            spot_dataframe = pd.concat(client.gather(spot_dataframe_futures))
            spot_dataframe.drop(labels=["frame"], axis=1, inplace=True)
            spot_dataframe.rename({"original_frame": "frame"}, axis=1, inplace=True)
        else:
            spot_dataframe = None

        if not keep_futures_spot_dataframe:
            del spot_dataframe_futures
            spot_dataframe_futures = None

        if not keep_futures_spots_movie:
            del spot_movie_futures
            spot_movie_futures = None

    else:
        spot_dataframe, _, _ = track_features.segmentation_df(
            spot_labels, spot_movie, frame_metadata, extra_properties=extra_properties
        )

        if drop_reverse_time:
            spot_dataframe.drop(
                labels=["frame_reverse", "t_frame_reverse"], axis=1, inplace=True
            )

        spot_dataframe_futures = None
        spot_labels_futures = None
        bandpassed_movie_futures = None
        spot_movie_futures = None

    if not return_spot_labels:
        del spot_labels
        spot_labels = None

    return (
        spot_dataframe,
        spot_dataframe_futures,
        spot_labels,
        spot_labels_futures,
        bandpassed_movie,
        bandpassed_movie_futures,
        spot_movie_futures,
    )


def _add_neighborhood_row(
    spot_dataframe_row,
    movie,
    span,
    pos_columns=["z", "y", "x"],
):
    """
    Extracts neighborhood specified by row in dataframe generated by
    :func:`~detect_spots`.
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
    Extracts neighborhood specified by each row in dataframe generated by
    :func:`~detect_spots`, adding the neighborhood to the dataframe in a
    "raw_spot" column.
    """
    spot_dataframe = spot_dataframe.copy()
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

    return spot_dataframe


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
    threshold_factor=1,
    min_size=0,
    connectivity=None,
    return_bandpass=False,
    keep_futures_bandpass=False,
    return_spot_labels=False,
    keep_futures_spot_labels=False,
    return_spot_dataframe=True,
    keep_futures_spot_dataframe=False,
    keep_futures_spots_movie=False,
    extra_properties=tuple(),
    drop_reverse_time=True,
    client=None,
):
    """
    Constructs a trackpy-compatible pandas Dataframe of proposed spot locations after
    bandpass-filtering, thresholding and removing small objects, adding a column for
    the raw spot data. If a Dask Client is passed as a `client` kwarg, the bandpass
    filtering and thresholding will be parallelized across the client.

    :param spot_movie: Movie of spot channel.
    :type spot_movie: np.ndarray
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
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :param span: Size of neighborhood to extract (rounded in each axis to the largest
        odd number below `span` if even).
    :type span: np.ndarray
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
    :param float threshold_factor: If using automated thresholding, this factor is multiplied
        by the proposed threshold value. This gives some degree of control over the stringency
        of thresholding while still getting a ballpark value using the automated method.
    :param int min_size: The smallest allowable object size.
    :param int connectivity: The connectivity defining the neighborhood of a pixel
        during small object removal.
    :param bool return_bandpass: If True, returns bandpass-filtered movie as
        third element of output. Otherwise returns none as third output.
    :param bool keep_futures_bandpass: If `True`, keeps generated bandpass-filtered movie
        as a list of `Futures` in the Dask worker memories, returning as a list in fourth output.
    :param bool return_spot_labels: If True, returns labelled spot mask as
        second element of output. Otherwise returns none as second output.
    :param bool keep_futures_spot_labels: If `True`, keeps generated labelled mask as a list of
        `Futures` in the Dask worker memories, returning as a list in second output.
    :param bool return_spot_dataframe: If True, returns fully evaluated dataframe of
        labelled spots as first output (otherwise returns `None`).
    :param bool keep_futures_spot_dataframe: If `True`, keeps dataframe of labelled
        spots as a list of `Futures` in the Dask worker memories.
    :param bool keep_futures_spots_movie: If `True`, keeps dataframe of input `spot_movie`
        as a list of `Futures` in the Dask worker memories.
    :param extra_properties: Properties of each labelled region in the segmentation
        mask to measure and add to the DataFrame. With no extra properties, the
        DataFrame will have columns only for the frame, label, and centroid
        coordinates.
    :type extra_properties: tuple[str]
    :param bool drop_reverse_time: If True, drops the columns with reversed frame
        numbers added to facilitate tracking (if you are using nuclear tracking to
        track your spots instead of tracking the spots independently, you might not
        need these columns).
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return:

        * trackpy-compatible pandas DataFrame of spot positions and extra requested
          properties.
        * trackpy-compatible pandas DataFrame of spot positions and extra requested
          properties, chunked up as list of `Futures` corresponding to labelled mask and
          bandpassed-filtered movie `Futures` (see below).
        * labelled mask of spots in `spot_movie`
        * labelled mask as list of `Futures`
        * bandpass-filtered movie
        * bandpass-filtered movie as a list of `Futures`
        * input `spot_movie` as list of `Futures`

    :rtype: tuple
    """
    if client is not None:
        detect_spots_return_spot_dataframe = False
        detect_spots_keep_futures = True
    else:
        detect_spots_return_spot_dataframe = True
        detect_spots_keep_futures = False

    (
        spot_dataframe,
        spot_dataframe_futures,
        spot_labels,
        spot_labels_futures,
        bandpassed_movie,
        bandpassed_movie_futures,
        spot_movie_futures,
    ) = detect_spots(
        spot_movie,
        frame_metadata=frame_metadata,
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        threshold=threshold,
        threshold_factor=threshold_factor,
        min_size=min_size,
        nbins=nbins,
        connectivity=connectivity,
        return_bandpass=return_bandpass,
        keep_futures_bandpass=keep_futures_bandpass,
        return_spot_labels=return_spot_labels,
        keep_futures_spot_labels=keep_futures_spot_labels,
        return_spot_dataframe=detect_spots_return_spot_dataframe,
        keep_futures_spot_dataframe=keep_futures_spot_dataframe,
        keep_futures_spots_movie=detect_spots_keep_futures,
        extra_properties=extra_properties,
        drop_reverse_time=drop_reverse_time,
        client=client,
    )

    if (spot_dataframe_futures is not None) and (spot_movie_futures is not None):

        def add_neighborhoods_func(spot_movie_chunk, spot_dataframe_chunk):
            neighborhoods_df = _add_neighborhoods_to_dataframe(
                spot_movie_chunk, span, spot_dataframe_chunk, pos_columns
            )
            return neighborhoods_df

        spot_dataframe_futures = client.map(
            add_neighborhoods_func, spot_movie_futures, spot_dataframe_futures
        )

        if return_spot_dataframe:
            spot_dataframe = pd.concat(client.gather(spot_dataframe_futures))
            spot_dataframe.drop(labels=["frame"], axis=1, inplace=True)
            spot_dataframe.rename({"original_frame": "frame"}, axis=1, inplace=True)
        else:
            spot_dataframe = None

        if not keep_futures_spots_movie:
            del spot_movie_futures
            spot_movie_futures = None

    else:
        spot_dataframe = _add_neighborhoods_to_dataframe(
            spot_movie, span, spot_dataframe, pos_columns
        )

    return (
        spot_dataframe,
        spot_dataframe_futures,
        spot_labels,
        spot_labels_futures,
        bandpassed_movie,
        bandpassed_movie_futures,
        spot_movie_futures,
    )
