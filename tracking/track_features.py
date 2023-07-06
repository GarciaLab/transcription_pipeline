from skimage.measure import regionprops_table
import pandas as pd
import trackpy as tp
from preprocessing import process_metadata
from utils.parallel_computing import send_compute_to_cluster


def _reverse_segmentation_df(segmentation_df):
    """
    Reverses the frame numbering in the segmentation dataframe to make trackpy track
    the movie backwards - this helps with structures with high acceleration by low
    deceleration, like nuclei as they divide. This adds a column with the reversed
    frame numbers in place to the input segmentation_df.
    """
    max_frame = segmentation_df.max(axis=0)["frame"]
    segmentation_df["frame_reverse"] = segmentation_df.apply(
        lambda row: int(1 + max_frame - row["frame"]),
        axis=1,
    )

    max_t_frame = segmentation_df.max(axis=0)["t_frame"]
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
        :func:``scikit.segmentation.watershed``.
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


def link_dataframe(
    segmentation_df,
    *,
    search_range,
    memory,
    pos_columns,
    t_column,
    velocity_predict=True,
    **kwargs
):
    """
    Use trackpy to link particles across frames, assigning unique particles an
    identity in a `particle` column added in place to the input segmentation_df
    dataframe.

    :param segmentation_df: trackpy-compatible pandas DataFrame for linking particles
        across frame.
    :type segmentation_df: pandas DataFrame
    :param float search_range: The maximum distance features can move between frames.
    :param int memory: The maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle.
    :param pos_columns: Name of columns in segmentation_df containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param t_column: Name of column in segmentation_df containing the frame number
        for each feature.
    :type t_column: DataFrame column name
    :param bool velocity_predict: If True, uses trackpy's
        `predict.NearestVelocityPredict` class to estimate a velocity for each feature
        at each timestep and predict its position in the next frame. This can help
        tracking, particularly of nuclei during nuclear divisions.
    :return: Original segmentation_df DataFrame with an added `particle` column
        assigning an ID to each unique feature as tracked by trackpy.
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

    # Reindex dataframe
    linked_dataframe = linked_dataframe.reset_index(drop=True)

    # Increment particle labels by 1 to avoid erasing 0-th particle
    linked_dataframe["particle"] = linked_dataframe["particle"].apply(lambda x: x + 1)

    return linked_dataframe


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
    for i, properties in linked_df.iterrows():
        frame_index = int(properties["frame"]) - 1
        old_label = properties["label"]
        new_label = properties["particle"]

        object = segmentation_mask[frame_index] == old_label
        reordered_mask[frame_index][object] = new_label

    return reordered_mask


def reorder_labels_parallel(segmentation_mask, linked_dataframe, **kwargs):
    """
    Relabels the input segmentation_mask to match the particle ID assigned by
    :func:`~link_dataframe`, with the relabeling operation parallelized across
    a Dask LocalCluster.

    :param segmentation_mask: A labeled array, with each label corresponding to a mask
        for a single feature.
    :type segmentation_mask: Numpy array.
    :param linked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
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
    :return: Segmentation mask for a movie with labels consistent between linked
        particles.
    :rtype: Numpy array.
    """
    reorder_labels_func = partial(reorder_labels, linked_dataframe)

    reordered_mask = send_compute_to_cluster(
        segmentation_mask, reorder_labels_func, kwargs, np.uint32
    )
    return reordered_mask