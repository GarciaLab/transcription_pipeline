from skimage.measure import regionprops_table
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import trackpy as tp
from ..preprocessing import process_metadata
from ..utils import parallel_computing
import scipy.signal as sig
import warnings
import logging
import multiprocessing
from tqdm import tqdm
import subprocess
import functools


def _number_detected_objects(feature_dataframe):
    """
    Construct an array of the number of detected objects for each frame.
    """
    frames = np.sort(feature_dataframe["frame"].unique())
    num_objects = np.array(
        [(feature_dataframe["frame"] == frame).sum() for frame in frames]
    )
    return frames, num_objects


def _norm_mean_nuclear_fluorescence(
    feature_dataframe, fluorescence_field="nuclear_intensity_mean"
):
    """
    Construct an array of the min-max normalized mean nuclear fluorescence in each
    frame.
    """
    frames = np.sort(feature_dataframe["frame"].unique())
    fluo = np.array(
        [
            feature_dataframe.loc[
                feature_dataframe["frame"] == frame, fluorescence_field
            ].mean()
            for frame in frames
        ]
    )
    norm_fluo = (fluo - fluo.min()) / (fluo.max() - fluo.min())
    return frames, norm_fluo


def determine_nuclear_cycle_frames(
    frame_array,
    property_array,
    *,
    trigger_property="num_objects",
    invert=False,
    **kwargs,
):
    """
    Assigns frames to nuclear divisions. Additional kwargs are passed through to the peak
    detection function `scipy.signal.find_peaks`.

    .. note::

        * If `trigger_property` is set to `num_objects`, the function looks for maxima of
          the discrete time-derivative of the log of the number of detected nuclei to assign
          frames to nuclear divisions. The log is used so that the same peak threshold
          parameters can be used for every nuclear cycle and any zoom dataset (doubling
          of the number of detected objects corresponding to a constant additive increase in
          log-space).
        * If `trigger_property` is set to `nuclear_fluorescence`, then the function
          goes off of increase (e.g. histone compaction) or decrease (loss of nuclear
          transcription factor) in nuclear fluorescence to estimate division frames, depending
          on the setting for `invert`.

    :param frame_array: Sorted array corresponding to the frame numbers.
    :type frame_array: np.ndarray
    :param property_array: Array corresponding to the image feature being used to determine
        the division frames - as of this version, this can number of detected features in
        each frame of `frame_array` (e.g. with a histone marker) or nuclear-localized
        fluorescence (e.g. with a fluorescently-tagged transcription factor).
    :type property_array: np.ndarray
    :param trigger_property: Image feature to use to determine the division frames.
    :type trigger_property: {'num_objects', 'nuclear_fluorescence'}
    :param bool invert: Invert `property_array` before using peak detection to mark
        division frames. This is useful e.g. if we're triggering off of loss of nuclear
        fluorescence during divisions. As of now this is only used if triggering off
        of nuclear fluorescence.
    :return: Array of the frames with detected division waves.
    :rtype: np.ndarray
    """
    property_array = np.copy(property_array)
    if trigger_property == "num_objects":
        # Using chain rule to take derivative of log of number of detected objects - this
        # makes the default parameters more generalizable to different zooms since
        # doubling of the number of objects corresponds to the same order-of-magnitude
        # peak in the derivative
        property_array = np.gradient(property_array) / property_array

    elif trigger_property == "nuclear_fluorescence":
        # Uses loss of nuclear fluorescence during nuclear divisions to estimate the
        # nuclear cycle frames
        if invert:
            property_array *= -1
            property_array -= property_array.min()

    else:
        raise Exception("`trigger_property` not recognized.")

    division_index, _ = sig.find_peaks(property_array, **kwargs)

    frames = frame_array[division_index]

    return frames


def _nuclear_cycle_by_number_objects(num_objects, num_nuclei_per_fov):
    """
    Determines the nuclear cycle based on the detected number of objects in the field
    of view. The acceptable range of detected number of objects per field of view
    is contained in the dictionary `num_nuclei_per_fov` in the form
    {nuclear_cycle: (lower bound, upper bound)}.
    """
    nuclear_cycle = []
    for cycle in num_nuclei_per_fov:
        if num_nuclei_per_fov[cycle][0] < num_objects < num_nuclei_per_fov[cycle][1]:
            nuclear_cycle.append(cycle)

    if len(nuclear_cycle) > 1:
        raise ValueError(
            "Range of number of objects in FOV used to determine nuclear cycle is overlapping."
        )

    if len(nuclear_cycle) == 0:
        warnings.warn(
            "Number of detected objects outside specified bounds for nuclear cycle determination.",
            stacklevel=2,
        )
        nuclear_cycle.append(np.nan)

    return nuclear_cycle[0]


def assign_nuclear_cycle(
    feature_dataframe,
    num_nuclei_per_fov,
    trigger_property="num_objects",
    invert=False,
    fluorescence_field="nuclear_intensity_mean",
    ignore_fluo_threshold=0.2,
    **kwargs,
):
    """
    Traverses input `feature_dataframe` (usually corresponding to a nuclear marker)
    of segmented features and uses the number of features per field-of-view to assign
    a nuclear cycle to the particle as per the specifications of a dictionary
    `num_nuclei_per_fov`. `feature_dataframe` is modified in-place to add a
    "nuclear_cycle" column. Additional kwargs are passed through to the peak
    detection function used to estimate division frames (see documentation for
    `determine_nuclear_cycle_frames`).

    .. note::

        Unless specified, a default parameter of `height` = 0.1, `distance` = 10 is
        passed through to `determine_nuclear_cycle_frames` if triggering off of the number
        of objects in the FOV. If triggering off of nuclear fluorescence, a default
        parameter of `prominence` = 0.5 is used instead for peak detection.

    :param feature_dataframe: Dataframe with each row corresponding to a detected and
        segmented feature, and a column `frame` containing the corresponding frame.
    :type feature_dataframe: pandas DataFrame
    :param dict num_nuclei_per_fov: Dictionary specifying the acceptable range of median
        number of detected objects per FOV for a contiguous series of frames bounded
        by detected division waves. The ranges are specified in the form
        {nuclear_cycle: (lower bound, upper bound)}.
    :param trigger_property: Image feature to use to determine the division frames. For
        more extensive description, see documentation for `determine_nuclear_cycle_frames`.
    :type trigger_property: {'num_objects', 'nuclear_fluorescence'}
    :param bool invert: Invert `property_array` before using peak detection to mark
        division frames. This is useful e.g. if we're triggering off of loss of nuclear
        fluorescence during divisions. As of now this is only used if triggering off
        of nuclear fluorescence.
    :param str fluorescence_field: Name of field used to quantify fluorescence for division
        frame assignment. Only used if `trigger_property` is set to "nuclear_fluorescence".
    :param float ignore_fluo_threshold: Fraction of peak nuclear fluorescence below
        which to ignore frames counting objects to avoid including spurious segmentation
        during nuclear divisions. Setting to 1 is equivalent to no threshold.
    :return: tuple(division_frames, nuclear_cycle)

        * `division_frames`: Numpy array of frame number (not index - the frames are
          1-indexed as per `trackpy`'s convention) of the detected division windows.
        * `nuclear_cycle`: Numpy array of nuclear cycle being exited at corresponding
          entry of `division_frames` - this will be one entry larger than `division_frames`
          since we obviously don't see division out of the last cycle observed.

    :rtype: tuple
    """
    frames, num_objects = _number_detected_objects(feature_dataframe)

    # Handle defaults for peak detection function
    if trigger_property == "num_objects":
        if "height" not in kwargs:
            kwargs["height"] = 0.1
        if "distance" not in kwargs:
            kwargs["distance"] = 10
        property_array = num_objects

    elif trigger_property == "nuclear_fluorescence":
        if "prominence" not in kwargs:
            kwargs["prominence"] = 0.5
        _, property_array = _norm_mean_nuclear_fluorescence(
            feature_dataframe, fluorescence_field
        )

        # If triggering off of nuclear fluorescence, we can only trust the
        # segmentation as long as frames with loss of fluorescence are excluded
        # when taking the object count.
        num_objects = num_objects.astype(float)
        ignore_frames = property_array < ignore_fluo_threshold * (property_array.max())
        num_objects[ignore_frames] = np.nan

    else:
        raise ValueError("`trigger_property` option not recognized.")

    division_frames = determine_nuclear_cycle_frames(
        frames,
        property_array,
        trigger_property=trigger_property,
        invert=invert,
        **kwargs,
    )

    division_indices = np.arange(frames.size)[np.isin(frames, division_frames)]
    division_split_frames = np.split(frames, division_indices)
    division_split_indices = np.split(np.arange(frames.size), division_indices)

    median_num_objects_cycle = [
        np.nanmedian(num_objects[cycle_indices])
        for cycle_indices in division_split_indices
    ]
    nuclear_cycle = [
        _nuclear_cycle_by_number_objects(num_objects, num_nuclei_per_fov)
        for num_objects in median_num_objects_cycle
    ]

    def _determine_nuclear_cycle(object_row):
        frame = object_row["frame"]
        for i, cycle_frames in enumerate(division_split_frames):
            if frame in cycle_frames:
                return nuclear_cycle[i]
        return None

    feature_dataframe["nuclear_cycle"] = feature_dataframe.apply(
        _determine_nuclear_cycle, axis=1
    )

    return division_frames, np.array(nuclear_cycle)


def _reverse_segmentation_df(segment_df):
    """
    Reverses the frame numbering in the segmentation dataframe to make trackpy track
    the movie backwards - this helps with structures with high acceleration by low
    deceleration, like nuclei as they divide. This adds a column with the reversed
    frame numbers in place to the input `segmentation_df`
    """
    max_frame = segment_df["frame"].max()
    segment_df["frame_reverse"] = segment_df.apply(
        lambda row: int(1 + max_frame - row["frame"]),
        axis=1,
    )

    max_t_frame = segment_df["t_frame"].max()
    reverse_t_frame = lambda row: (
        np.nan if np.isnan(row["t_frame"]) else int(max_t_frame - row["t_frame"])
    )
    segment_df["t_frame_reverse"] = segment_df.apply(
        reverse_t_frame,
        axis=1,
    )
    return None


def segmentation_df(
    segmentation_mask,
    intensity_image,
    frame_metadata,
    *,
    initial_frame_index=0,
    num_nuclei_per_fov=None,
    fluorescence_field="nuclear_intensity_mean",
    extra_properties=tuple(),
    extra_properties_callable=None,
    spacing=None,
    **kwargs,
):
    """
    Constructs a trackpy-compatible pandas DataFrame for tracking from a
    frame-indexed array of segmentation masks. Additional kwargs are passed through
    to `assign_nuclear_cycle`.

    :param segmentation_mask: Integer-labelled segmentation, as returned by
        `scikit.segmentation.watershed`.
    :type segmentation_mask: np.ndarray[np.int]
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :param int initial_frame_index: Index of first frame, used to offset the recorded
        frame numbers. This is useful if execution is being parallelized by mapping onto
        chunks of a movie.
    :param intensity_image: Intensity (i.e., input) image with same size as
        labeled image.
    :type intensity_image: np.ndarray
    :param dict num_nuclei_per_fov: Dictionary specifying the acceptable range of median
        number of detected objects per FOV for a contiguous series of frames bounded
        by detected division waves. The ranges are specified in the form
        {nuclear_cycle: (lower bound, upper bound)}. This can be passed as `None` (Default)
        to skip assigning nuclear cycles.
    :param str fluorescence_field: Name of field used to quantify fluorescence for division
        frame assignment. Only used if `trigger_property` is set to "nuclear_fluorescence".
    :param extra_properties: Properties of each labelled region in the segmentation
        mask to measure and add to the DataFrame. With no extra properties, the
        DataFrame will have columns only for the frame, label, and centroid
        coordinates.
    :type extra_properties: tuple[str]
    :param extra_properties_callable: Properties of each labelled region in the segmentation
        mask to measure and add to the DataFrame, using an iterable of callables for properties
        not included with skimage.
    :type extra_properties_callable: Iterable of callables, optional.
    :param spacing: The pixel spacing along each axis of the image. `None` defaults to
        isotropic.
    :type spacing: tuple of float
    :return: tuple(mitosis_dataframe, division_frames, nuclear_cycle) where

        * `mitosis_dataframe`: pandas DataFrame of frame, label, centroids, and imaging time
          `t_s` for each labelled region in the segmentation mask (along with other measurements
          specified by extra_properties). Also includes column `t_frame` for the imaging time
          in units of z-stack scanning time, and columns `frame_reverse` and `t_frame_reverse`
          with the frame numbers reversed to allow tracking in reverse (this performs better
          on high-acceleration, low-deceleration particles), along with the assigned nuclear
          cycle for the particle as per `assign_nuclear_cycle`.
        * `division_frames`: Numpy array of frame number (not index - the frames are
          1-indexed as per `trackpy`'s convention) of the detected division windows.
        * `nuclear_cycle`: Numpy array of nuclear cycle being exited at corresponding
          entry of `division_frames` - this will be one entry larger than `division_frames`
          since we obviously don't see division out of the last cycle observed.

    :rtype: tuple[pd.DataFrame, np.ndarray, np.ndarray]
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
                extra_properties=extra_properties_callable,
                spacing=spacing,
            )
            num_labels = np.unique(frame_properties["label"]).size
            frame_properties["frame"] = np.full(num_labels, i + 1)
            frame_properties["original_frame"] = (
                frame_properties["frame"] + initial_frame_index
            )

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

    # Add imaging time for each particle. The if-else statement is required to avoid
    # errors due to single-pixel labels that throw np.nan when the centroid is requested.
    time = process_metadata.extract_time(frame_metadata)[0]
    time_apply = lambda row: (
        np.nan if np.isnan(row["z"]) else time(int(row["original_frame"]), row["z"])
    )
    movie_properties["t_s"] = movie_properties.apply(time_apply, axis=1)

    # Add imaging time in number of frames for each particles.
    time_frame = process_metadata.extract_renormalized_frame(frame_metadata)
    time_frame_apply = lambda row: (
        np.nan
        if np.isnan(row["z"])
        else time_frame(int(row["original_frame"]), row["z"])
    )
    movie_properties["t_frame"] = movie_properties.apply(time_frame_apply, axis=1)

    # Add columns with reversed frame and t_frame to enable reverse tracking
    _reverse_segmentation_df(movie_properties)

    # Add nuclear cycles for later use when compiling traces
    if num_nuclei_per_fov is not None:
        division_frames, nuclear_cycle = assign_nuclear_cycle(
            movie_properties,
            num_nuclei_per_fov,
            fluorescence_field=fluorescence_field,
            **kwargs,
        )
    else:
        division_frames = None
        nuclear_cycle = None

    return movie_properties, division_frames, nuclear_cycle


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
    added columns for the instantaneous velocities in coordinates specified in
    `pos_columns`.

    :param tracked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type tracked_dataframe: pandas DataFrame
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


def read_trackpy_status(file_path, num_frames):
    """
    Helper function to read the trackpy progress file. Redirects trackpy's
    logging to a specified file that is then monitored in `tqdm`.

    :param str file_path: Path to file that trackpy's logging is being
        redirected to.
    :param int num_frames: Total number of frames trackpy is tracking
        over.
    """
    tail = subprocess.Popen(
        ["tail", "-F", file_path], stdout=subprocess.PIPE, bufsize=1, text=True
    )
    frame_iter = 0
    progress_bar_init = False
    for _ in iter(tail.stdout.readline, ""):
        if not progress_bar_init:
            progress_bar = tqdm(total=(num_frames - 1), desc="Tracking")
            progress_bar_init = True
        frame_iter += 1
        # noinspection PyUnboundLocalVariable
        progress_bar.update(1)
        if frame_iter == (num_frames - 1):
            tail.kill()
            progress_bar.close()


def tqdm_trackpy(
    linking_function, *, num_frames=None, trackpy_log_path="/tmp/trackpy_log"
):
    """
    A wrapper around `trackpy` linkers that redirects tracking progress to a file path
    and monitors it on a `tqdm` progress bar.

    :param linking_function: `trackpy` linking function.
    :type linking_function: function
    :param int num_frames: Total number of frames in dataframe being linked.
    :param str trackpy_log_path: Path to log file to redirect trackpy's stdout progress to.
    """

    # Identify `trackpy`'s logger for the tracking info messages printed
    # to console.
    @functools.wraps(linking_function)
    def linking_decorator(*args, **kwargs):
        trackpy_logger = logging.root.manager.loggerDict["trackpy"]

        # Set `trackpy`'s logger to quiet.
        trackpy_handler = trackpy_logger.handlers[0]
        trackpy_handler.setLevel(logging.WARN)

        # Add our own handler to read the tracking info messages
        f = open(trackpy_log_path, "w")
        temp_handler = logging.StreamHandler(f)
        temp_handler.setLevel(logging.INFO)
        trackpy_logger.addHandler(temp_handler)

        with multiprocessing.Pool(processes=1) as pool:
            pool.apply_async(read_trackpy_status, args=(trackpy_log_path, num_frames))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not generate velocity field for prediction: no tracks",
                )
                linking_result = linking_function(*args, **kwargs)
            pool.close()
            pool.join()

        return linking_result

    return linking_decorator


def link_df(
    segmentation_dataframe,
    *,
    search_range,
    memory,
    pos_columns,
    t_column,
    velocity_predict=True,
    velocity_averaging=None,
    add_velocities=True,
    reindex=True,
    monitor_progress=True,
    trackpy_log_path="/tmp/trackpy_log",
    **kwargs,
):
    """
    Use trackpy to link particles across frames, assigning unique particles an
    identity in a `particle` column added in place to the input segmentation_df
    dataframe and an estimate of the instantaneous velocity of tracked features
    in each coordinate (this expects the coordinate columns to be given as
    `x`, `y` and `z` and if those cannot be found will fall back on coordinates
    specified in `pos_columns`).

    :param segmentation_dataframe: trackpy-compatible pandas DataFrame for linking particles
        across frame.
    :type segmentation_dataframe: pandas DataFrame
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
    :param bool add_velocities: If True, adds velocities of tracked particles after
        tracking.
    :param bool reindex: If `True`, reindexes the dataframe after linking. This is
        important for subsequent mitosis detection on the nuclear channel, but
        prevents the split-apply-combine operations on the dataframe during spot
        analysis.
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
    if velocity_predict:
        pred = tp.predict.NearestVelocityPredict()
        link = pred.link_df
    else:
        link = tp.link_df

    # Occasionally some connected components will be a single-pixel thick in one of the
    # coordinates, which throws `np.nan` as a centroid in that coordinate. We handle this
    # by screening against `np.nan` in the dataframe before tracking.
    finite_position = segmentation_dataframe[pos_columns].notnull().all(axis=1)
    finite_time = segmentation_dataframe[t_column].notnull()

    finite_filtered_dataframe = segmentation_dataframe[
        finite_position & finite_time
    ].copy()

    if monitor_progress:
        num_frames = finite_filtered_dataframe[t_column].nunique()
        link = tqdm_trackpy(
            link, num_frames=num_frames, trackpy_log_path=trackpy_log_path
        )

    linked_dataframe = link(
        finite_filtered_dataframe,
        search_range=search_range,
        memory=memory,
        pos_columns=pos_columns,
        t_column=t_column,
        **kwargs,
    )

    # Reindex dataframe, next step in mitosis detection needs fresh index
    if reindex:
        linked_dataframe = linked_dataframe.reset_index(drop=True)

    # Increment particle labels by 1 to avoid erasing 0-th particle
    linked_dataframe["particle"] = (linked_dataframe["particle"] + 1).astype(int)

    #  Try to add velocities for all coordinates for flexibility, if not fall
    # back to velocities in requested coordinates.
    if add_velocities:
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

    label_mask = segmentation_mask[frame_index] == old_label
    reordered_mask[frame_index][label_mask] = new_label
    return reordered_mask


def reorder_labels(segmentation_mask, linked_dataframe):
    """
    Relabels the input segmentation_mask to match the particle ID assigned by
    :func:`~link_dataframe`.

    :param segmentation_mask: A labeled array, with each label corresponding to a mask
        for a single feature.
    :type segmentation_mask: np.ndarray
    :param linked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
    :return: Segmentation mask for a movie with labels consistent between linked
        particles.
    :rtype: np.ndarray
    """
    reordered_mask = np.zeros(segmentation_mask.shape, dtype=segmentation_mask.dtype)
    # Remove filtered particles to avoid unnecessary switches
    filtered_linked_dataframe = linked_dataframe[linked_dataframe["particle"] != 0]

    # Switch labels using 'particle' column in linked dataframe
    filtered_linked_dataframe.apply(
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


def reorder_labels_parallel(
    segmentation_mask, linked_dataframe, client, first_last_frames=None, **kwargs
):
    """
    Relabels the input segmentation_mask to match the particle ID assigned by
    :func:`~link_dataframe`, with the relabeling operation parallelized across
    a Dask LocalCluster.

    :param segmentation_mask: A labeled array, with each label corresponding to a mask
        for a single feature.
    :type segmentation_mask: {np.ndarray, list}
    :param linked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
    :param list first_last_frames: 2D list, with each 2-element along axis 0 corresponding
        to the first and last indices of the corresponding chunk in `segmentation_mask`.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: tuple(`reordered_labels`, `reordered_labels_futures`, `scattered_movies`)
        where

        * `reordered_labels` is the fully evaluated segmentation mask reordered as per
          the tracking, as an ndarray of the same shape as and dtype as
          `segmentation_mask`, with unique integer labels corresponding to each nucleus.
        * `reordered_labels_futures` is the list of futures objects resulting from the
          reordering in the worker memories before gathering and concatenation.
        * `scattered_data` is a list with each element corresponding to a list of
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
    if first_last_frames is None:
        num_processes = len(client.scheduler_info()["workers"])
        num_frames = parallel_computing.number_of_frames(segmentation_mask, client)
        frame_array = np.arange(num_frames) + 1  # Frames are 1-indexed
        split_array = np.array_split(frame_array, num_processes)

        first_last_frames = []
        for chunk in split_array:
            first_last_frames.append([chunk[0], chunk[-1]])

        warnings.warn(
            "".join(
                [
                    "No `first_last_frames` options passed, ",
                    "defaulting to chunking by number of available workers.",
                ]
            )
        )

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
