from ..utils import label_manipulation, parallel_computing
from pathlib import Path
import numpy as np
from scipy import stats
from ..tracking import track_features
from ..tracking.track_features import _reverse_segmentation_df
import zarr
import dask.dataframe as dd
import warnings


def _transfer_nuclear_labels_row(spot_dataframe_row, nuclear_labels, pos_columns):
    """
    Uses a provided nuclear mask to transfer the nuclear label of the nucleus
    containing the detected spot in a row of the spot dataframe output by
    :func"`~spot_analysis.detection`.
    """
    t_coordinate = spot_dataframe_row["frame"] - 1
    spatial_coordinates = spot_dataframe_row[pos_columns].astype(float).round().values
    coordinates = np.array([t_coordinate, *spatial_coordinates], dtype=int)
    label = nuclear_labels[(*coordinates,)]
    return label


def transfer_nuclear_labels(
    spot_dataframe,
    nuclear_labels,
    expand_distance=1,
    pos_columns=["z", "y", "x"],
    working_memory_mode=None,
    working_memory_folder=None,
    client=None,
):
    """
    Uses a provided nuclear mask to transfer the nuclear label of the nucleus
    containing each detected spot in the spot dataframe as output by
    :func"`~spot_analysis.detection`. If a `client` argument is passed, `nuclear_mask`
    may also be given as a list of futures as per the conventions of
    `utils.parallel_computing` for parallelization across a Dask LocalCluster. The input
    `spot_dataframe` is modified in-place to add a "particle" column with the
    corresponding labels.

    :param spot_dataframe: DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe: pandas DataFrame
    :param nuclear_labels: Labelled movie of nuclear masks.
    :type nuclear_labels: Numpy array of integers
    :param int expand_distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :param pos_columns: Name of columns in `spot_dataframe` containing a pixel-space
        position coordinate to be used to map to the nuclear labels.
    :type pos_columns: list of DataFrame column names
    :param working_memory_mode: Sets whether the intermediate steps that need to be
        evaluated to construct the nuclear tracking (e.g. construction of a nuclear
        segmentation array) are kept in memory or committed to a `zarr` array. Note
        that using `zarr` as working memory requires the input data to also be a
        `zarr` array, whereby the chunking is inferred from the input data.
    :type working_memory: {"zarr", None}
    :param working_memory_folder: This parameter is required if `working_memory`
        is set to `zarr`, and should be a folder path that points to the location
        where the necessary `zarr` arrays will be stored to disk.
    :type working_memory_folder: {str, `pathlib.Path`, None}
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: None
    :rtype: None
    """
    # Expand nuclear labels to spots at the very surface
    if client is None:
        expanded_labels = label_manipulation.expand_labels_movie(
            nuclear_labels, distance=expand_distance
        )
    else:
        if working_memory_mode == "zarr":
            evaluate_expansion = False
            futures_expansion = True

            working_memory_path = Path(working_memory_folder)
            results_path = working_memory_path / "expanded_labels"
            results_path.mkdir(exist_ok=True)

            expanded_labels_zarr = zarr.creation.zeros_like(
                nuclear_labels,
                overwrite=True,
                store=results_path / "expanded_labels.zarr",
            )

            zarr_in_mode = isinstance(nuclear_labels, zarr.core.Array)

            if zarr_in_mode:
                chunk_boundaries, labels = parallel_computing.zarr_to_futures(
                    nuclear_labels, client
                )

            else:
                temp_labels = zarr.creation.array(
                    nuclear_labels,
                    overwrite=True,
                    store=results_path / "nuclear_labels.zarr",
                )

                chunk_boundaries, labels = parallel_computing.zarr_to_futures(
                    temp_labels, client
                )

        elif working_memory_mode is None:
            evaluate_expansion = True
            futures_expansion = False

            labels = nuclear_labels

        else:
            raise ValueError("`working_memory_mode` option not recognized.")

        (
            expanded_labels,
            expanded_labels_futures,
            _,
        ) = label_manipulation.expand_labels_movie_parallel(
            labels,
            distance=expand_distance,
            client=client,
            futures_in=False,
            futures_out=futures_expansion,
            evaluate=evaluate_expansion,
        )

        if working_memory_mode == "zarr":
            parallel_computing.futures_to_zarr(
                expanded_labels_futures,
                chunk_boundaries,
                expanded_labels_zarr,
                client,
            )

            expanded_labels = expanded_labels_zarr

    # Transfer labels from expanded nuclear mask to each spot
    def _transfer_labels_df(spot_df, *, expanded_labels_array):
        """
        Helper function to tranfer labels from expanded label array.
        """
        nuclear_labels_series = spot_df.apply(
            _transfer_nuclear_labels_row,
            args=(
                expanded_labels_array,
                pos_columns,
            ),
            axis=1,
        )
        return nuclear_labels_series

    if client is None:
        spot_dataframe["nuclear_label"] = _transfer_labels_df(
            spot_dataframe, expanded_labels_array=expanded_labels
        )
    else:
        if working_memory_mode != "zarr":
            warnings.warn(
                "`working_memory_mode` is not zarr, labeled array will be serialized to workers."
            )
        # Convert to dask dataframe and partition across processes
        num_processes = len(client.scheduler_info()["workers"])
        spot_dataframe_dask = dd.from_pandas(spot_dataframe, npartitions=num_processes)
        spot_dataframe["nuclear_label"] = spot_dataframe_dask.map_partitions(
            _transfer_labels_df,
            expanded_labels_array=expanded_labels,
            meta=(None, "i4"),
        ).compute()

    return None


def filter_spots_by_sigma(
    spot_dataframe, *, sigma_x_y_bounds=(0, np.inf), sigma_z_bounds=(0, np.inf)
):
    """
    Adds a column of booleans `include_spot_by_sigma` marking spots for inclusion
    based on the fit sigmas falling within specified ranges `sigma_x_y_bounds` and
    `sigma_z_bounds` for `sigma_x_y` and `sigma_z` respectively. The input
    `spot_dataframe` is modified in-place.

    :param spot_dataframe: DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe: pandas DataFrame
    :param sigma_x_y_bounds: Tuple(sigma_x_y lower bound, sigma_x_y upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_x_y_bounds: Tuple of floats.
    :param sigma_z_bounds: Tuple(sigma_z lower bound, sigma_z upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_z_bounds: Tuple of floats.
    :return: None
    :rtype: None
    .. note::
        This also automatically excludes any points that couldn't be fitted.
    """
    sigma_x_y_lb, sigma_x_y_ub = sigma_x_y_bounds
    sigma_z_lb, sigma_z_ub = sigma_z_bounds

    include_spots_x_y = (spot_dataframe["sigma_x_y"] > sigma_x_y_lb) & (
        spot_dataframe["sigma_x_y"] < sigma_x_y_ub
    )
    include_spots_z = (spot_dataframe["sigma_z"] > sigma_z_lb) & (
        spot_dataframe["sigma_z"] < sigma_z_ub
    )

    include_spots = include_spots_x_y & include_spots_z

    spot_dataframe["include_spot_by_sigma"] = include_spots

    return None


def track_spots(
    spot_dataframe,
    *,
    search_range,
    memory,
    pos_columns,
    t_column,
    velocity_predict=True,
    velocity_averaging=None,
    **kwargs,
):
    """
    Tracks spots in input `spot_dataframe` using trackpy, renaming any previous
    `particle` labels (e.g. from transfer of nuclear labels to the spots) to
    `particle_nuclear` and adding a `particle` column with new labels from trackpy
    linking.

    :param segmentation_df: trackpy-compatible pandas DataFrame for linking particles
        across frame.
    :type segmentation_df: pandas DataFrame
    :param float search_range: The maximum distance features can move between frames.
    :param int memory: The maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle.
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param t_column: Name of column in `segmentation_df` containing the time coordinate
        for each feature.
    :type t_column: DataFrame column name,
        {`frame`, `t_frame`, `frame_reverse`, `t_frame_reverse`}. For explanation of
        column names, see :func:`~segmentation_df`.
    :param bool velocity_predict: If True, uses trackpy's
        `predict.NearestVelocityPredict` class to estimate a velocity for each feature
        at each timestep and predict its position in the next frame. This can help
        tracking, particularly of nuclei during nuclear divisions.
    :param int averaging: Number of frames to average velocity over.
    :return: Original `segmentation_df` DataFrame with an added `particle` column
        assigning an ID to each unique feature as tracked by trackpy and velocity
        columns for each coordinate in `pos_columns`.
    :rtype: pandas DataFrame

    .. note::
        This function can also take any kwargs accepted by ``trackpy.link_df`` to
        specify the tracking options.
    """
    # Add reversed-time coordinates in case it helps with tracking
    if (t_column == "frame_reverse") or (t_column == "t_frame_reverse"):
        _reverse_segmentation_df(spot_dataframe)

    # Rename any previous tracking from nuclear label transfer to prevent overwrite
    spot_dataframe.rename(columns={"particle": "previous_tracking"}, errors="ignore")

    # Track using trackpy
    linked_spot_dataframe = track_features.link_df(
        spot_dataframe,
        search_range=search_range,
        memory=memory,
        pos_columns=pos_columns,
        t_column=t_column,
        velocity_predict=velocity_predict,
        velocity_averaging=velocity_averaging,
        reindex=False,
        **kwargs,
    )

    linked_spot_dataframe.rename(
        columns={"particle": "trackpy_label"}, errors="raise", inplace=True
    )

    return linked_spot_dataframe


def _not_stub(tracked_spot_dataframe_row, tracked_spot_dataframe, min_track_length):
    """
    Returns `True` if a given track has over `min_track_length` points in its trackpy
    label, `False` otherwise.
    """
    length = (
        tracked_spot_dataframe["trackpy_label"]
        == tracked_spot_dataframe_row["trackpy_label"]
    ).sum()
    not_stub = length > min_track_length
    return not_stub


def filter_spots_by_tracks(tracked_spot_dataframe, min_track_length):
    """
    Adds a column of booleans `include_spot_by_track` marking spots for inclusion
    based on whether a given spot could be tracked for more than `min_track_length`
    points. The input `spot_dataframe` is modified in-place.

    :param spot_dataframe: DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe: pandas DataFrame
    :param int min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis.
    :return: None
    :rtype: None
    """
    tracked_spot_dataframe["include_spot_by_track"] = tracked_spot_dataframe.apply(
        _not_stub, args=(tracked_spot_dataframe, min_track_length), axis=1
    )
    return None


def _remove_duplicate_row(spot_dataframe_row, spot_dataframe, *, choose_by, min_or_max):
    """
    Checks input `spot_dataframe` for spots with the same nuclear label in the same
    frame, and marks each row with `True` if it is the only spot in that nucleus or if
    it is the spot with the best discriminant value (as set by `choose_by` and
    `min_or_max` - for instance, we can choose for spots with the highest amplitude of
    the Gaussian fit in the dataframe by setting `choose_by = "amplitude"` and
    `min_or_max = "maximize"`).
    """
    spots_nuclear_mask = (spot_dataframe["frame"] == spot_dataframe_row["frame"]) & (
        spot_dataframe["nuclear_label"] == spot_dataframe_row["nuclear_label"]
    )

    discriminant_value = spot_dataframe_row[choose_by]
    if min_or_max == "minimize":
        best_discriminant = (spot_dataframe.loc[spots_nuclear_mask, choose_by]).min()
    elif min_or_max == "maximize":
        best_discriminant = (spot_dataframe.loc[spots_nuclear_mask, choose_by]).max()
    else:
        raise ValueError("`min_or_max` option not recognized.")

    if discriminant_value == best_discriminant:
        choose_row = True
    else:
        choose_row = False

    return choose_row


def filter_multiple_spots(spot_dataframe, *, choose_by, min_or_max):
    """
    Checks input `spot_dataframe` for spots with the same nuclear label in the same
    frame, and marks each row with `True` in an added `include_spot_from_multiple`
    column if it is the only spot in that nucleus or if it is the spot with the best
    discriminant value (as set by `choose_by` and `min_or_max` - for instance, we can
    choose for spots with the highest amplitude of the Gaussian fit in the dataframe by
    setting `choose_by = "amplitude"` and `min_or_max = "maximize"`). Modifies the input
    `spot_dataframe` in-place.

    :param spot_dataframe: DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe: pandas DataFrame
    :param str choose_by: Name of column in `spot_dataframe` whose values to use
        as discriminating factor when choosing which of multiple spots in a nucleus
        to keep.
    :param min_or_max: Sets whether the spots are chosen so as to maximize or minimize
        the value in the column prescribed by `choose_by`.
    :type min_or_max: {"minimize", "maximize"}
    :return: None
    :rtype: None
    """
    spot_dataframe["include_spot_from_multiple"] = spot_dataframe.apply(
        _remove_duplicate_row,
        args=(spot_dataframe,),
        **{"choose_by": choose_by, "min_or_max": min_or_max},
        axis=1,
    )
    return None


def _compile_trace(
    tracked_dataframe,
    particle,
    min_points=5,
    quantification="intensity_from_neighborhood",
):
    """Returns an array of successive differences in intensity traces."""
    # Compile trace
    trace_dataframe = tracked_dataframe[
        tracked_dataframe["particle"] == particle
    ].copy()
    trace_dataframe = trace_dataframe.sort_values("t_s")

    # Filter stubs
    if trace_dataframe.index.size < min_points:
        return np.array([])

    # Calculate successive differences across trace
    intensity_series = trace_dataframe[quantification]
    differences = np.ediff1d(intensity_series.values)

    return differences


def compile_successive_differences(
    tracked_dataframe, min_points=5, quantification="intensity_from_neighborhood"
):
    """
    Compiles as a single array all successive differences in quantitation
    values of tracked traces.
    :param tracked_dataframe: DataFrame containing information about detected,
        filtered and tracked spots.
    :type tracked_dataframe: pandas DataFrame
    :param int min_points: Minimum number of data points for a trace to be
        considered when compiling successive differences.
    :param str quantification: Name of dataframe column containing quantification to use
        when compiling successive differences along a trace. Defaults to
        `intensity_from_neighborhood`.
    :return: Pooled array of all successive differences along all traces.
    :rtype: Numpy array.
    """
    # Find all unique particles
    particles = np.sort(np.trim_zeros(tracked_dataframe["particle"].unique()))

    # Compile all successive differences
    differences = np.array([])
    for particle in particles:
        differences = np.append(
            differences,
            _compile_trace(
                tracked_dataframe,
                particle,
                min_points=min_points,
                quantification=quantification,
            ),
        )

    return differences


def successive_differences_quartile(
    tracked_dataframe, min_points=5, quantification="intensity_from_neighborhood"
):
    """
    Estimates the .84-quantile of successive differences in intensity across
    all tracked traces. We pick .84 arbitrarily as a robust estimator of
    the standard deviation - this can be used downstream to set an "exchange
    rate" between spatial proximity and similarity in intensity when tracking
    spots.

    :param tracked_dataframe: DataFrame containing information about detected,
        filtered and tracked spots.
    :type tracked_dataframe: pandas DataFrame
    :param int min_points: Minimum number of data points for a trace to be
        considered when compiling successive differences.
    :param str quantification: Name of dataframe column containing quantification to use
        when compiling successive differences along a trace. Defaults to
        `intensity_from_neighborhood`.
    :return: Pooled array of all successive differences along all traces.
    :rtype: Numpy array
    """
    all_differences = compile_successive_differences(
        tracked_dataframe, min_points=min_points, quantification=quantification
    )
    quantile = np.quantile(all_differences, 0.84)
    return quantile


def normalized_variation_intensity(
    tracked_dataframe,
    normalize_quantile_to=1,
    min_points=5,
    quantification="intensity_from_neighborhood",
):
    """
    Given a preliminary spot tracking, this normalizes the intensity quantification of
    the spots so that the .84-quantile  of the successive difference in intensity
    across all tracked traces is `normalize_quantile_to`. This can be used to set the
    weight assigned to intensity when re-tracking (i.e. intensity is added as an extra
    spatial dimension so that spots with wild jumps in intensity are less likely to be
    linked). Note that the .84-quantile is chosen arbitrarily as a robust estimator
    of the standard deviation of a normal distribution.

    :param tracked_dataframe: DataFrame containing information about detected,
        filtered and tracked spots.
    :type tracked_dataframe: pandas DataFrame
    :param float normalize_quantile_to_1: Target value of .84-quantile of successive
        differences in intensity across traces after normalization. This can essentially
        be used to set the "exchange rate" between spatial proximity and similarity in
        intensity (i.e. when tracking, differing in intensity by the .84-quantile
        is penalized as much as being separated by `normalize_quantile_to` from a
        candidate point in the next frame).
    :param int min_points: Minimum number of data points for a trace to be
        considered when compiling successive differences.
    :param str quantification: Name of dataframe column containing quantification to use
        when compiling successive differences along a trace. Defaults to
        `intensity_from_neighborhood`.
    :return: Normalized intensity as per function description above.
    :rtype: pandas Series
    """
    quantile = successive_differences_quartile(
        tracked_dataframe, min_points=min_points, quantification=quantification
    )

    normalized_intensity = (
        tracked_dataframe[quantification] / quantile
    ) * normalize_quantile_to

    return normalized_intensity


def track_and_filter_spots(
    spot_dataframe,
    *,
    sigma_x_y_bounds=(0, np.inf),
    sigma_z_bounds=(0, np.inf),
    nuclear_labels=None,
    expand_distance=1,
    search_range=10,
    memory=2,
    pos_columns=["z", "y", "x"],
    nuclear_pos_columns=["z", "y", "x"],
    t_column="frame_reverse",
    velocity_predict=True,
    velocity_averaging=None,
    min_track_length=3,
    filter_multiple=True,
    choose_by="intensity_from_neighborhood",
    min_or_max="maximize",
    filter_negative=True,
    quantification="intensity_from_neighborhood",
    track_by_intensity=False,
    normalize_quantile_to=1,
    min_track_length_intensity=5,
    client=None,
    working_memory_mode=None,
    working_memory_folder=None,
    **kwargs,
):
    """
    Traverses input `spot_dataframe` of detected and fitted spots and excludes spots with
    fit standard deviations outside of specified ranges. If an array of tracked nuclear
    labels is provided, extranuclear spots are excluded as well, and the remaining spots are
    tracked using `trackpy`. Spots corresponding to short, spurious traces are excluded.
    If an array of tracked nuclear labels is provided, the nuclear labels are expanded
    and transferred over to the remaining spots. If there are multiple spots remaining
    in a single nucleus at any point, one is chosen by a specified characteristic (e.g.
    to maximize amplitude of the Gaussian fit or minimize the normalized cost). Particles
    excluded at any stage are assigned a `particle` label of 0. The input `spot_dataframe`
    is modified in-place to add a `particle` column.

    :param spot_dataframe: DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe: pandas DataFrame
    :param sigma_x_y_bounds: Tuple(sigma_x_y lower bound, sigma_x_y upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_x_y_bounds: Tuple of floats.
    :param sigma_z_bounds: Tuple(sigma_z lower bound, sigma_z upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_z_bounds: Tuple of floats.
    :param nuclear_labels: Labelled movie of nuclear masks.
    :type nuclear_labels: Numpy array of integers
    :param int expand_distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :param float search_range: The maximum distance features can move between frames.
    :param int memory: The maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle.
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate, used to track particles using `trackpy`.
    :type pos_columns: list of DataFrame column names
    :param nuclear_pos_columns: Name of columns in `spot_dataframe` containing a pixel-space
        position coordinate to be used to map to the nuclear labels.
    :type nuclear_pos_columns: list of DataFrame column names
    :param t_column: Name of column in `segmentation_df` containing the time coordinate
        for each feature.
    :type t_column: DataFrame column name,
        {`frame`, `t_frame`, `frame_reverse`, `t_frame_reverse`}. For explanation of
        column names, see :func:`~segmentation_df`.
    :param bool velocity_predict: If True, uses trackpy's
        `predict.NearestVelocityPredict` class to estimate a velocity for each feature
        at each timestep and predict its position in the next frame. This can help
        tracking, particularly of nuclei during nuclear divisions.
    :param int averaging: Number of frames to average velocity over.
    :param int min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis.
    :param bool filter_multiple: Decide whether or not to enforce a single-spot
        limit for each nucleus.
    :param str choose_by: Name of column in `spot_dataframe` whose values to use
        as discriminating factor when choosing which of multiple spots in a nucleus
        to keep.
    :param min_or_max: Sets whether the spots are chosen so as to maximize or minimize
        the value in the column prescribed by `choose_by`.
    :type min_or_max: {"minimize", "maximize"}
    :param bool filter_negative: Ignores datapoints where the spot quantification
        goes negative - as long as we are looking at background-subtracted intensity,
        negative values are clear mistrackings/misquantifications.
    :param str quantification: Name of dataframe column containing quantification to use
        if filtering by negatives or filtering multiple nuclear spots by intensity.
        Defaults to `intensity_from_neighborhood`.
    :param bool track_by_intensity: If `True`, this will attempt to use a preliminary
        spot tracking to use the variation of intensity across traces to help spot
        tracking (i.e. spots with wild jumps in intensity are less likely to be linked
        across frames). See documentation for `normalized_variation_intensity` for
        details.
    :param float normalize_quantile_to: Target value of .84-quantile of successive
        differences in intensity across traces after normalization. This can essentially
        be used to set the "exchange rate" between spatial proximity and similarity in
        intensity (i.e. when tracking, differing in intensity by the .84-quantile
        is penalized as much as being separated by `normalize_quantile_to` from a
        candidate point in the next frame) - the higher the value use, the less tolerant
        tracking is of large variations in intensity.
    :param int min_track_length_intensity: Minimum number of data points for a trace to be
        considered when compiling successive differences for intensity-based retracking.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param working_memory_mode: Sets whether the intermediate steps that need to be
        evaluated to construct the nuclear tracking (e.g. construction of a nuclear
        segmentation array) are kept in memory or committed to a `zarr` array. Note
        that using `zarr` as working memory requires the input data to also be a
        `zarr` array, whereby the chunking is inferred from the input data.
    :type working_memory: {"zarr", None}
    :param working_memory_folder: This parameter is required if `working_memory`
        is set to `zarr`, and should be a folder path that points to the location
        where the necessary `zarr` arrays will be stored to disk.
    :type working_memory_folder: {str, `pathlib.Path`, None}
    :return: None
    :rtype: None

    .. note::
        This function can also take any kwargs accepted by ``trackpy.link_df`` to
        specify the tracking options.
    """
    spot_df = spot_dataframe.copy()

    # First filter out spots with unphysical sigmas
    filter_spots_by_sigma(
        spot_df, sigma_x_y_bounds=sigma_x_y_bounds, sigma_z_bounds=sigma_z_bounds
    )
    spot_df = spot_df[spot_df["include_spot_by_sigma"]]

    # If quantification is available, we can filter out spots that have a negative
    # intensity since those are mistrackings/misquantifications.
    if filter_negative:
        try:
            positive_filter = spot_df[quantification] > 0
            spot_df = spot_df[positive_filter]
        except KeyError:
            warnings.warn(
                "".join(
                    [
                        "Could not find requested quantification column, ",
                        "skipping filtering negative-intensity spots.",
                    ]
                )
            )

    # Transfer nuclear labels if provided
    if nuclear_labels is not None:
        transfer_nuclear_labels(
            spot_df,
            nuclear_labels,
            expand_distance=expand_distance,
            pos_columns=nuclear_pos_columns,
            working_memory_mode=working_memory_mode,
            working_memory_folder=working_memory_folder,
            client=client,
        )

        # Make subdataframe excluding extranuclear spots
        spot_df = spot_df[~(spot_df["nuclear_label"] == 0)]

    # Add normalized intensity if intensity-based tracking is requested
    retrack_pos_columns = pos_columns.copy()
    if track_by_intensity:
        try:
            normalized_intensity = normalized_variation_intensity(
                spot_df,
                normalize_quantile_to=normalize_quantile_to,
                min_points=min_track_length_intensity,
                quantification=quantification,
            )

            spot_df["normalized_intensity"] = normalized_intensity
            retrack_pos_columns.append("normalized_intensity")

        except KeyError:
            warnings.warn(
                "".join(
                    [
                        "Could not normalize intensity for retracking, ",
                        "check previous tracking and intensity quantification.",
                    ]
                )
            )

    # Track remaining spots
    tracked_spots_df = track_spots(
        spot_df.copy(),
        search_range=search_range,
        memory=memory,
        pos_columns=retrack_pos_columns,
        t_column=t_column,
        velocity_predict=velocity_predict,
        velocity_averaging=velocity_averaging,
        **kwargs,
    )

    # Filter out short spurious tracks
    filter_spots_by_tracks(tracked_spots_df, min_track_length)
    filtered_tracked_spots_df = tracked_spots_df[
        tracked_spots_df["include_spot_by_track"]
    ].copy()

    # Handle remaining multiple spots in same nucleus
    if (nuclear_labels is not None) and filter_multiple:
        filter_multiple_spots(
            filtered_tracked_spots_df, choose_by=choose_by, min_or_max=min_or_max
        )
        filtered_tracked_spots_df = filtered_tracked_spots_df[
            filtered_tracked_spots_df["include_spot_from_multiple"]
        ].copy()

    # Add labels back to original dataframe
    spot_dataframe["particle"] = 0
    if nuclear_labels is not None:
        spot_dataframe.loc[
            filtered_tracked_spots_df.index, "particle"
        ] = filtered_tracked_spots_df["nuclear_label"]
    else:
        spot_dataframe.loc[
            filtered_tracked_spots_df.index, "particle"
        ] = filtered_tracked_spots_df["trackpy_label"]

    return None