from skimage.measure import regionprops_table
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import trackpy as tp
from preprocessing import process_metadata
from functools import partial
import dask
from utils import parallel_computing


def _reverse_segmentation_df(segmentation_df):
    """
    Reverses the frame numbering in the segmentation dataframe to make trackpy track
    the movie backwards - this helps with structures with high acceleration by low
    deceleration, like nuclei as they divide. This adds a column with the reversed
    frame numbers in place to the input `segmentation_df`
    """
    max_frame = segmentation_df["frame"].max()
    segmentation_df["frame_reverse"] = segmentation_df.apply(
        lambda row: int(1 + max_frame - row["frame"]),
        axis=1,
    )

    max_t_frame = segmentation_df["t_frame"].max()
    segmentation_df["t_frame_reverse"] = segmentation_df.apply(
        lambda row: int(max_t_frame - row["t_frame"]),
        axis=1,
    )
    return None


def segmentation_df(
    segmentation_mask, intensity_image, frame_metadata, *, extra_properties=tuple()
):
    """
    Constructs a trackpy-compatible pandas DataFrame for tracking from a
    frame-indexed array of segmentation masks.

    :param segmentation_mask: Integer-labelled segmentation, as returned by
        `scikit.segmentation.watershed`.
    :type segmentation_mask: Numpy array of integers.
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :param intensity_image: Intensity (i.e., input) image with same size as
        labeled image.
    :type intensity_image: Numpy array.
    :param extra_properties: Properties of each labelled region in the segmentation
        mask to measure and add to the DataFrame. With no extra properties, the
        DataFrame will have columns only for the frame, label, and centroid
        coordinates.
    :type extra_properties: Tuple of strings, optional.
    :param str z_label: Axis label corresponding to z-axis, used to interpolate
        time between z-slices if necessary.
    :return: pandas DataFrame of frame, label, centroids, and imaging time `t_s` for
        each labelled region in the segmentation mask (along with other measurements
        specified by extra_properties). Also includes column `t_frame` for the imaging
        time in units of z-stack scanning time, and columns `frame_reverse` and
        `t_frame_reverse` with the frame numbers reversed to allow tracking in reverse
        (this performs better on high-acceleration, low-deceleration particles).
    :rtype: pandas DataFrame
    """
    # Go over every frame and make a pandas-compatible dict for each labelled object
    # in the segmentation.
    movie_properties = []

    num_timepoints = segmentation_mask.shape[0]
    for i in range(num_timepoints):
        if segmentation_mask[i].any():
            frame_properties = regionprops_table(
                segmentation_mask[i],
                intensity_image=intensity_image[i],
                properties=("label", "centroid_weighted") + extra_properties,
            )
            num_labels = np.unique(frame_properties["label"]).size
            frame_properties["frame"] = np.full(num_labels, i + 1)

            frame_properties = pd.DataFrame.from_dict(frame_properties)
            movie_properties.append(frame_properties)

    movie_properties = pd.concat(movie_properties)
    movie_properties = movie_properties.reset_index(drop=True)  # Reset index of rows

    # Rename centroid columns
    num_dim_frame = segmentation_mask.ndim - 1
    rename_columns = {}
    spatial_axes = "zyx"
    for i in range(num_dim_frame):
        old_column_name = "".join(["centroid_weighted-", str(i)])
        new_column_name = spatial_axes[i]
        rename_columns[old_column_name] = new_column_name

    movie_properties.rename(rename_columns, axis=1, inplace=True)

    # Add imaging time for each particle
    time = process_metadata.extract_time(frame_metadata)[0]
    time_apply = lambda row: time(int(row["frame"]), row["z"])
    movie_properties["t_s"] = movie_properties.apply(time_apply, axis=1)

    # Add imaging time in number of frames for each particles
    time_frame = process_metadata.extract_renormalized_frame(frame_metadata)
    time_frame_apply = lambda row: time_frame(int(row["frame"]), row["z"])
    movie_properties["t_frame"] = movie_properties.apply(time_frame_apply, axis=1)

    # Add columns with reversed frame and t_frame to enable reverse tracking
    _reverse_segmentation_df(movie_properties)

    return movie_properties


def _calculate_velocities(particle_dataframe, pos, t_column, averaging):
    """
    Returns an array of velocities in the single coordinate `pos` for each frame in a
    particle-grouped dataframe after tracking by calculating successive differences.
    """
    if particle_dataframe.shape[0] == 1:
        velocity_series = np.array([np.nan])
    else:
        time_sorted_dataframe = particle_dataframe.sort_values(t_column)
        positions_series = time_sorted_dataframe[pos].values
        time_series = time_sorted_dataframe[t_column].values
        velocity_series = np.zeros(positions_series.size)

        # Calculate successive differences
        delta_position = -np.diff(positions_series)

        # Find number of timepoints between each sample
        delta_t = np.diff(time_series)

        velocity_series[1:] = delta_position / delta_t
        velocity_series[0] = velocity_series[1]  # Reflective boundary condition

        if averaging is not None:
            velocity_series = uniform_filter1d(
                velocity_series, size=averaging, mode="mirror"
            )

    return velocity_series


def add_velocity(tracked_dataframe, pos_columns, t_column, averaging=None):
    """
    Returns a copy of a dataframe of tracked features (post linking with trackpy) with
    added columns for the instantaneous velocities in cooridinates specified in
    `pos_columns`.
    :param tracked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param t_column: Name of column in `segmentation_df` containing the time-coordinate
        for each feature.
    :type t_column: DataFrame column name,
        {`frame`, `t_frame`, `frame_reverse`, `t_frame_reverse`}. For explanation of
        column names, see :func:`~segmentation_df`.
    :param int averaging: Number of frames to average velocity over.
    :return: Copy of input tracked dataframe with added columns for coordinate
        velocities.
    :rtype: pandas DataFrame
    """
    dataframe = tracked_dataframe.copy()
    vel_columns = ["".join(["v_", pos]) for pos in pos_columns]
    dataframe[vel_columns] = np.nan

    # Construct array of particle labels
    particles = np.unique(dataframe["particle"].values)

    for particle in particles:
        particle_subdataframe_index = dataframe["particle"] == particle

        for i, vel in enumerate(vel_columns):
            dataframe.loc[particle_subdataframe_index, vel] = _calculate_velocities(
                dataframe.loc[particle_subdataframe_index],
                pos_columns[i],
                t_column,
                averaging=averaging,
            )
    return dataframe


def link_df(
    segmentation_df,
    *,
    search_range,
    memory,
    pos_columns,
    t_column,
    velocity_predict=True,
    velocity_averaging=None,
    **kwargs
):
    """
    Use trackpy to link particles across frames, assigning unique particles an
    identity in a `particle` column added in place to the input segmentation_df
    dataframe and an estimate of the instantaneous velocity of tracked features
    in each coordinate (this expects the coordinate columns to be given as
    `x`, `y` and `z` and if those cannot be found will fall back on coordinates
    specified in `pos_columns`).

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
    if velocity_predict:
        pred = tp.predict.NearestVelocityPredict()
        link = pred.link_df
    else:
        link = tp.link_df

    linked_dataframe = link(
        segmentation_df,
        search_range=search_range,
        memory=memory,
        pos_columns=pos_columns,
        t_column=t_column,
        **kwargs,
    )

    # Increment particle labels by 1 to avoid erasing 0-th particle
    linked_dataframe["particle"] = linked_dataframe["particle"].apply(lambda x: x + 1)

    #  Try to add velocities for all coordinates for flexibility, if not fall
    # back to velocities in requested coordinates.
    vel_columns = ["z", "y", "x"]
    try:
        linked_dataframe = add_velocity(
            linked_dataframe, vel_columns, t_column, averaging=velocity_averaging
        )
    except KeyError:
        linked_dataframe = add_velocity(
            linked_dataframe, pos_columns, t_column, averaging=velocity_averaging
        )

    # Drop reversed time coordinates since we're done with trackpy
    linked_dataframe.drop(
        ["frame_reverse", "t_frame_reverse"], axis=1, inplace=True, errors="ignore"
    )

    return linked_dataframe


def _switch_labels(properties, segmentation_mask, reordered_mask):
    """
    Reorders segmentation mask labels specified by a row `properties` of a linked
    segmentation dataframe to match the tracking. The output reordered segmentation
    array reordered_mask is passed as an argument and modified in-place.
    """
    frame_index = int(properties["frame"]) - 1
    old_label = properties["label"]
    new_label = properties["particle"]

    object = segmentation_mask[frame_index] == old_label
    reordered_mask[frame_index][object] = new_label
    return reordered_mask


def reorder_labels(segmentation_mask, linked_dataframe):
    """
    Relabels the input segmentation_mask to match the particle ID assigned by
    :func:`~link_dataframe`.

    :param segmentation_mask: A labeled array, with each label corresponding to a mask
        for a single feature.
    :type segmentation_mask: Numpy array.
    :param linked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
    :return: Segmentation mask for a movie with labels consistent between linked
        particles.
    :rtype: Numpy array.
    """
    reordered_mask = np.zeros(segmentation_mask.shape, dtype=segmentation_mask.dtype)

    # Switch labels using 'particle' column in linked dataframe
    linked_dataframe.apply(
        _switch_labels, args=(segmentation_mask, reordered_mask), axis=1
    )

    return reordered_mask


def _chunk_dataframe(dataframe, first_frame, last_frame):
    """
    Select [`first_frame`, `last_frame`] from a movie and reindex the `frames`
    column so that they are indexed within the chunk starting at `first_frame`.
    """
    chunk_selector = dataframe["frame"].between(first_frame, last_frame)
    chunk_dataframe = dataframe[chunk_selector].copy()

    chunk_dataframe["frame"] = chunk_dataframe["frame"].apply(
        lambda x: x - first_frame + 1
    )

    return chunk_dataframe


def reorder_labels_parallel(segmentation_mask, linked_dataframe, client, **kwargs):
    """
    Relabels the input segmentation_mask to match the particle ID assigned by
    :func:`~link_dataframe`, with the relabeling operation parallelized across
    a Dask LocalCluster.

    :param segmentation_mask: A labeled array, with each label corresponding to a mask
        for a single feature.
    :type segmentation_mask: Numpy array of integers or list of Futures corresponding
        to chunks of `segmentation_mask`.
    :param linked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: Tuple(`reordered_labels`, `reordered_labels_futures`, `scattered_movies`)
        where
        *`reordered_labels` is the fully evaluated segmentation mask reordered as per
        the tracking, as an ndarray of the same shape as and dtype as
        `segmentation_mask`, with unique integer labels corresponding to each nucleus.
        *`reordered_labels_futures` is the list of futures objects resulting from the
        reordering in the worker memories before gathering and concatenation.
        *`scattered_data` is a list with each element corresponding to a list of
        futures pointing to the input `segmentation_mask` and `linked_dataframe` in
        the workers' memory respectively.
    :rtype: tuple
    .. note::
        This function can also pass along any kwargs taken by
        :func:`~utils.parallel_computing.parallelize`.
    """
    evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
        kwargs
    )

    # Figure out which frames to split `linked_dataframe` around.
    num_processes = len(client.scheduler_info()["workers"])
    num_frames = parallel_computing.number_of_frames(segmentation_mask, client)
    frame_array = np.arange(num_frames) + 1  # Frames are 1-indexed
    split_array = np.array_split(frame_array, num_processes)

    first_last_frames = []
    for chunk in split_array:
        first_last_frames.append([chunk[0], chunk[-1]])

    split_linked_dataframe = []
    for frame_split in first_last_frames:
        split_linked_dataframe.append(
            _chunk_dataframe(linked_dataframe, frame_split[0], frame_split[1])
        )

    # Manually scatter split dataframes to pass type check in `parallelize`.
    scattered_linked_dataframe = client.scatter(split_linked_dataframe)

    (
        reordered_labels,
        reordered_labels_futures,
        scattered_data,
    ) = parallel_computing.parallelize(
        [segmentation_mask, scattered_linked_dataframe],
        reorder_labels,
        client,
        evaluate=evaluate,
        futures_in=futures_in,
        futures_out=futures_out,
    )

    return reordered_labels, reordered_labels_futures, scattered_data