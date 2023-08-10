from ..utils import label_manipulation
import numpy as np
from ..tracking import track_features
from ..tracking.track_features import _reverse_segmentation_df


def _transfer_nuclear_labels_row(spot_dataframe_row, nuclear_labels):
    """
    Uses a provided nuclear mask to transfer the nuclear label of the nucleus
    containing the detected spot in a row of the spot dataframe output by
    :func"`~spot_analysis.detection`.
    """
    t_coordinate = spot_dataframe_row["frame"] - 1
    spatial_coordinates = (
        spot_dataframe_row[["z", "y", "x"]].astype(float).round().values
    )
    coordinates = np.array([t_coordinate, *spatial_coordinates], dtype=int)
    label = nuclear_labels[(*coordinates,)]
    return label


def transfer_nuclear_labels(
    spot_dataframe, nuclear_labels, expand_distance=1, client=None
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
    spot_dataframe["nuclear_label"] = spot_dataframe.apply(
        _transfer_nuclear_labels_row, args=(expanded_labels,), axis=1
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


def track_and_filter_spots(
    spot_dataframe,
    *,
    sigma_x_y_bounds=(0, np.inf),
    sigma_z_bounds=(0, np.inf),
    nuclear_labels=None,
    expand_distance=1,
    search_range=10,
    memory=2,
    pos_columns=["y", "x"],
    t_column="frame_reverse",
    velocity_predict=True,
    velocity_averaging=None,
    min_track_length=3,
    choose_by="amplitude",
    min_or_max="maximize",
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
    :param int min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis.
    :param str choose_by: Name of column in `spot_dataframe` whose values to use
        as discriminating factor when choosing which of multiple spots in a nucleus
        to keep.
    :param min_or_max: Sets whether the spots are chosen so as to maximize or minimize
        the value in the column prescribed by `choose_by`.
    :type min_or_max: {"minimize", "maximize"}
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
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

    # Transfer nuclear labels if provided
    if nuclear_labels is not None:
        transfer_nuclear_labels(
            spot_df, nuclear_labels, expand_distance=expand_distance, client=client
        )

        # Make subdataframe excluding extranuclear spots
        spot_df = spot_df[~(spot_df["nuclear_label"] == 0)]

    # Track remaining spots
    tracked_spots_df = track_spots(
        spot_df.copy(),
        search_range=search_range,
        memory=memory,
        pos_columns=pos_columns,
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
    if nuclear_labels is not None:
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