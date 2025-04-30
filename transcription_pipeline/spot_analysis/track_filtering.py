from ..utils import label_manipulation
import numpy as np
import warnings
from ..tracking import track_features

# noinspection PyProtectedMember
from ..tracking.track_features import _reverse_segmentation_df
from tqdm import tqdm


def _transfer_nuclear_labels_row(
    spot_dataframe_row, nuclear_labels, pos_columns, frame_column
):
    """
    Uses a provided nuclear mask to transfer the nuclear label of the nucleus
    containing the detected spot in a row of the spot dataframe output by
    :func"`~spot_analysis.detection`.
    """
    t_coordinate = spot_dataframe_row[frame_column] - 1
    spatial_coordinates = spot_dataframe_row[pos_columns].astype(float).round().values
    coordinates = np.array([t_coordinate, *spatial_coordinates], dtype=int)
    label = nuclear_labels[(*coordinates,)]
    return label


def transfer_nuclear_labels(
    spot_dataframe,
    nuclear_labels,
    expand_distance=1,
    pos_columns=["z", "y", "x"],
    frame_column="frame",
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
    :type nuclear_labels: np.ndarray[np.int]
    :param int expand_distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :param pos_columns: Name of columns in `spot_dataframe` containing a pixel-space
        position coordinate to be used to map to the nuclear labels.
    :type pos_columns: list of DataFrame column names
    :param str frame_column: Name of column containing the frame numbers. This is provided
        as an option to facilitate working directly with dataframe Futures generate from
        chunked movies before stitching together in the local process.
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
        expanded_labels, _, _ = label_manipulation.expand_labels_movie_parallel(
            nuclear_labels,
            distance=expand_distance,
            client=client,
            futures_in=False,
            futures_out=False,
            evaluate=True,
        )

    # Transfer labels from expanded nuclear mask to each spot
    # No progress bar here since this can be handled inside
    # Dask Futures.
    spot_dataframe["nuclear_label"] = spot_dataframe.apply(
        _transfer_nuclear_labels_row,
        args=(
            expanded_labels,
            pos_columns,
            frame_column,
        ),
        axis=1,
    )

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
    :param sigma_x_y_bounds: tuple(sigma_x_y lower bound, sigma_x_y upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_x_y_bounds: tuple[float]
    :param sigma_z_bounds: tuple(sigma_z lower bound, sigma_z upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_z_bounds: tuple[float]
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
    monitor_progress=True,
    trackpy_log_path="/tmp/trackpy_log",
    **kwargs,
):
    """
    Tracks spots in input `spot_dataframe` using trackpy, renaming any previous
    `particle` labels (e.g. from transfer of nuclear labels to the spots) to
    `particle_nuclear` and adding a `particle` column with new labels from trackpy
    linking.

    :param spot_dataframe: trackpy-compatible pandas DataFrame for linking particles
        across frame.
    :type spot_dataframe: pandas DataFrame
    :param float search_range: The maximum distance features can move between frames.
    :param int memory: The maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle.
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
    :param int velocity_averaging: Number of frames to average velocity over.
    :param bool monitor_progress: If True, redirects the output of `trackpy`'s
        tracking monitoring to a `tqdm` progress bar.
    :param str trackpy_log_path: Path to log file to redirect trackpy's stdout progress to.
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
        add_velocities=False,
        monitor_progress=monitor_progress,
        trackpy_log_path=trackpy_log_path,
        **kwargs,
    )

    linked_spot_dataframe.rename(
        columns={"particle": "trackpy_label"}, errors="raise", inplace=True
    )

    return linked_spot_dataframe


def filter_spots_by_tracks(tracked_spot_dataframe, min_track_length):
    """
    Adds a column of booleans `include_spot_by_track` marking spots for inclusion
    based on whether a given spot could be tracked for more than `min_track_length`
    points. The input `spot_dataframe` is modified in-place.

    :param tracked_spot_dataframe: DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type tracked_spot_dataframe: pandas DataFrame
    :param int min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis.
    :return: None
    :rtype: None
    """
    if min_track_length < 1:
        tracked_spot_dataframe["include_spot_by_track"] = True
        return None

    track_lengths = tracked_spot_dataframe.groupby(
        "trackpy_label", as_index=False, sort=False
    ).size()

    tracked_spot_dataframe["include_spot_by_track"] = (
        tracked_spot_dataframe.merge(
            track_lengths,
            on="trackpy_label",
        )["size"]
        > min_track_length
    ).values

    return None


def _remove_duplicate_row(
    spot_dataframe_row, spot_dataframe, *, choose_by, min_or_max, max_num_spots=1
):
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

    if min_or_max == "minimize":
        best_discriminant_idx = (
            (spot_dataframe.loc[spots_nuclear_mask, choose_by])
            .nsmallest(max_num_spots)
            .index
        )
    elif min_or_max == "maximize":
        best_discriminant_idx = (
            (spot_dataframe.loc[spots_nuclear_mask, choose_by])
            .nlargest(max_num_spots)
            .index
        )
    else:
        raise ValueError("`min_or_max` option not recognized.")

    if spot_dataframe_row.name in best_discriminant_idx:
        choose_row = True
    else:
        choose_row = False

    return choose_row


def filter_multiple_spots(spot_dataframe, *, choose_by, min_or_max, max_num_spots=1):
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
    :param int max_num_spots: Maximum number of allowed spots per nuclear label, if a
        `nuclear_labels` is provided.
    :return: None
    :rtype: None
    """
    tqdm.pandas(desc="Filtering multiple spots in nuclear labels")
    spot_dataframe["include_spot_from_multiple"] = spot_dataframe.progress_apply(
        _remove_duplicate_row,
        args=(spot_dataframe,),
        **{
            "choose_by": choose_by,
            "min_or_max": min_or_max,
            "max_num_spots": max_num_spots,
        },
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
    :rtype: np.ndarray
    """
    # Find all unique particles
    particles = np.sort(np.trim_zeros(tracked_dataframe["particle"].unique()))
    num_particles = particles.size

    # Compile all successive differences
    differences = np.array([])
    for particle in tqdm(
        particles,
        total=num_particles,
        desc="Compiling variations in intensity",
    ):
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
    :rtype: np.ndarray
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
    :param float normalize_quantile_to: Target value of .84-quantile of successive
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
    frame_column="frame",
    nuclear_pos_columns=["z", "y", "x"],
    t_column="frame_reverse",
    velocity_predict=True,
    velocity_averaging=None,
    min_track_length=3,
    choose_by="intensity_from_neighborhood",
    min_or_max="maximize",
    max_num_spots=1,
    filter_negative=True,
    quantification="intensity_from_neighborhood",
    track_by_intensity=False,
    normalize_quantile_to=1,
    min_track_length_intensity=5,
    monitor_progress=True,
    trackpy_log_path="/tmp/trackpy_log",
    verbose=False,
    client=None,
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
    :param sigma_x_y_bounds: tuple(sigma_x_y lower bound, sigma_x_y upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_x_y_bounds: tuple[float]
    :param sigma_z_bounds: tuple(sigma_z lower bound, sigma_z upper bound)
        setting the acceptable range for inclusion of spots, with spots falling outside
        that ranged being marked with a `False` value in the added `include_spot_by_sigma`
        column.
    :type sigma_z_bounds: tuple[float]
    :param nuclear_labels: Labelled movie of nuclear masks.
    :type nuclear_labels: {np.ndarray[np.int], None}
    :param int expand_distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :param float search_range: The maximum distance features can move between frames.
    :param int memory: The maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle.
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate, used to track particles using `trackpy`.
    :type pos_columns: list of DataFrame column names
    :param nuclear_pos_columns: Name of columns in `spot_dataframe` containing a pixel-space
        position coordinate to be used to map to the nuclear labels.
    :param str frame_column: Name of column containing the frame numbers. This is provided
        as an option to facilitate working directly with dataframe Futures generate from
        chunked movies before stitching together in the local process - in this case
        it is used to transfer nuclear labels when the whole labeled array has been passed
        instead of corresponding chunks.
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
    :param int velocity_averaging: Number of frames to average velocity over.
    :param int min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis.
    :param str choose_by: Name of column in `spot_dataframe` whose values to use
        as discriminating factor when choosing which of multiple spots in a nucleus
        to keep.
    :param min_or_max: Sets whether the spots are chosen so as to maximize or minimize
        the value in the column prescribed by `choose_by`.
    :type min_or_max: {"minimize", "maximize"}
    :param int max_num_spots: Maximum number of allowed spots per nuclear label, if a
        `nuclear_labels` is provided.
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
    :param bool monitor_progress: If True, redirects the output of `trackpy`'s
        tracking monitoring to a `tqdm` progress bar.
    :param str trackpy_log_path: Path to log file to redirect trackpy's stdout progress to.
    :param bool verbose: If `True`, marks each row of the spot dataframe with the boolean
        flag indicating where the spot may have been filtered out.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: None
    :rtype: None

    .. note::
        This function can also take any kwargs accepted by ``trackpy.link_df`` to
        specify the tracking options.
    """
    # Transfer nuclear labels if provided
    try:
        # Make subdataframe excluding extranuclear spots
        spot_df = spot_dataframe[~(spot_dataframe["nuclear_label"] == 0)].copy()
    except KeyError:
        if nuclear_labels is not None:
            transfer_nuclear_labels(
                spot_dataframe,
                nuclear_labels,
                expand_distance=expand_distance,
                pos_columns=nuclear_pos_columns,
                frame_column=frame_column,
                client=client,
            )
            spot_df = spot_dataframe[~(spot_dataframe["nuclear_label"] == 0)].copy()
        else:
            spot_df = spot_dataframe.copy()

    with tqdm(total=2, desc="Preliminary spot filtering") as pbar:
        # First filter out spots with unphysical sigmas
        filter_spots_by_sigma(
            spot_df, sigma_x_y_bounds=sigma_x_y_bounds, sigma_z_bounds=sigma_z_bounds
        )
        if verbose:
            spot_dataframe["include_spot_by_sigma"] = False
            spot_dataframe.loc[spot_df.index, "include_spot_by_sigma"] = spot_df[
                "include_spot_by_sigma"
            ]

        spot_df = spot_df[spot_df["include_spot_by_sigma"]]
        pbar.update(1)

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
        pbar.update(1)

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
        monitor_progress=monitor_progress,
        trackpy_log_path=trackpy_log_path,
        **kwargs,
    )

    if verbose:
        if "trackpy_id" in spot_dataframe:
            spot_dataframe["previous_trackpy_id"] = spot_dataframe["trackpy_id"].copy()

        spot_dataframe["trackpy_id"] = 0
        spot_dataframe.loc[tracked_spots_df.index, "trackpy_id"] = tracked_spots_df[
            "trackpy_label"
        ]

    with tqdm(total=2, desc="Post-tracking spot filtering") as pbar:
        # Filter out short spurious tracks
        filter_spots_by_tracks(tracked_spots_df, min_track_length)

        if verbose:
            spot_dataframe["include_spot_by_track"] = False
            spot_dataframe.loc[tracked_spots_df.index, "include_spot_by_track"] = (
                tracked_spots_df["include_spot_by_track"]
            )
            try:
                spot_dataframe["normalized_intensity"] = np.nan
                spot_dataframe.loc[tracked_spots_df.index, "normalized_intensity"] = (
                    tracked_spots_df["normalized_intensity"]
                )
            except KeyError:
                spot_dataframe.drop("normalized_intensity", axis=1, inplace=True)

        filtered_tracked_spots_df = tracked_spots_df[
            tracked_spots_df["include_spot_by_track"]
        ].copy()
        pbar.update(1)

        # Handle remaining multiple spots in same nucleus
        if nuclear_labels is not None:
            filter_multiple_spots(
                filtered_tracked_spots_df,
                choose_by=choose_by,
                min_or_max=min_or_max,
                max_num_spots=max_num_spots,
            )

            if verbose:
                spot_dataframe.loc[
                    filtered_tracked_spots_df.index, "include_spot_from_multiple"
                ] = filtered_tracked_spots_df["include_spot_from_multiple"]

            filtered_tracked_spots_df = filtered_tracked_spots_df[
                filtered_tracked_spots_df["include_spot_from_multiple"]
            ].copy()
        pbar.update(1)

    # Add labels back to original dataframe
    spot_dataframe["particle"] = 0
    spot_dataframe["particle"] = spot_dataframe["particle"].astype(int)

    if nuclear_labels is not None:
        spot_dataframe["nuclear_label"] = 0
        spot_dataframe["nuclear_label"] = spot_dataframe["nuclear_label"].astype(int)

        spot_dataframe.loc[filtered_tracked_spots_df.index, "nuclear_label"] = (
            filtered_tracked_spots_df["nuclear_label"]
        )

        if max_num_spots == 1:
            spot_dataframe.loc[filtered_tracked_spots_df.index, "particle"] = (
                filtered_tracked_spots_df["nuclear_label"]
            )

        else:
            spot_dataframe.loc[filtered_tracked_spots_df.index, "particle"] = (
                filtered_tracked_spots_df["trackpy_label"]
            )

    else:
        spot_dataframe.loc[filtered_tracked_spots_df.index, "particle"] = (
            filtered_tracked_spots_df["trackpy_label"]
        )

    return None
