from . import fitting
from . import compile_data
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm


def _compile_traces_background_averaging(
    tracked_spots_dataframe,
    *,
    min_frames,
    pos_columns=["z", "y", "x"],
    compile_columns=[],
    nuclear_tracking_dataframe=None,
    compile_columns_nuclear=["nuclear_cycle"],
    ignore_negative_spots=True,
):
    """ """
    filtered_dataframe = tracked_spots_dataframe[
        tracked_spots_dataframe["particle"] != 0
    ].copy()

    filtered_compiled_data = compile_data.compile_traces(
        filtered_dataframe,
        compile_columns_spot=[
            "frame",
            "t_s",
            "raw_spot",
            *pos_columns,
            "coordinates_start",
            *compile_columns,
        ],
        nuclear_tracking_dataframe=nuclear_tracking_dataframe,
        compile_columns_nuclear=compile_columns_nuclear,
        ignore_negative_spots=ignore_negative_spots,
    )

    # Restrict to longer traces, filtering stubs.
    filtered_compiled_traces = filtered_compiled_data[
        filtered_compiled_data["frame"].apply(lambda x: x.size) > min_frames
    ].copy()
    filtered_compiled_traces.reset_index(inplace=True, drop=True)

    def _compile_centroids(row):
        return (
            np.vstack(row[pos_columns].to_numpy()).T
            - np.array([*row["coordinates_start"]])[:, 1:]
        )

    filtered_compiled_traces["centroid"] = None
    filtered_compiled_traces["centroid"] = filtered_compiled_traces["centroid"].astype(
        object
    )
    filtered_compiled_traces["centroid"] = filtered_compiled_traces.apply(
        _compile_centroids, axis=1
    )

    drop_columns = [*pos_columns, "coordinates_start"]
    drop_columns = [column for column in drop_columns if column not in compile_columns]

    filtered_compiled_traces.drop(drop_columns, axis=1, inplace=True)

    return filtered_compiled_traces


def _collect_background_pixels(
    trace_raw_spots,
    t_s,
    trace_t_s,
    trace_centroids,
    *,
    window,
    win_type,
    std,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
):
    """
    Helper function to collect background pixels in a time window for
    intensity quantification.
    """
    background_pixels = []
    if win_type == "gaussian":
        background_pixels_weights = []
    elif win_type == "boxcar":
        background_pixels_weights = None
    else:
        raise NotImplementedError("Unsupported window type {}".format(win_type))

    window_mask = ((t_s - (window / 2)) < trace_t_s) & (
        trace_t_s <= (t_s + (window / 2))
    )

    # Time difference for gaussian weights
    if background_pixels_weights is not None:
        time_diff = trace_t_s[window_mask] - t_s
        # Downstream function enforces normalization
        window_weights = np.exp(-0.5 * (time_diff / std) ** 2)
    else:
        window_weights = None

    # Grab raw spots
    window_raw_spots = trace_raw_spots[window_mask]
    window_centroids = trace_centroids[window_mask]
    for j in range(window_mask.sum()):
        _, frame_background_pixels = fitting.extract_spot_shell(
            window_raw_spots[j],
            centroid=window_centroids[j],
            mppZ=mppZ,
            mppYX=mppYX,
            ball_diameter_um=ball_diameter_um,
            shell_width_um=shell_width_um,
            aspect_ratio=aspect_ratio,
        )
        background_pixels.append(frame_background_pixels)

        if background_pixels_weights is not None:
            frame_background_pixels_weights = window_weights[j] * np.ones_like(
                frame_background_pixels, dtype=float
            )
            background_pixels_weights.append(frame_background_pixels_weights)

    background_pixels = np.concatenate(background_pixels)
    if background_pixels_weights is not None:
        background_pixels_weights = np.concatenate(background_pixels_weights)

    return background_pixels, background_pixels_weights


def refine_trace(
    trace_raw_spots,
    trace_t_s,
    trace_centroids,
    *,
    window,
    win_type,
    std,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
    num_bootstraps,
    background,
):
    """
    Improves the quantification of a trace by averaging over a few frames to estimate
    the spot background, while still estimating spot intensity on a frame-by-frame basis.
    This should only be used for traces with sampling times much shorter than timescales
    over which the background varies significantly (~30s in preliminary tests).

    :param trace_raw_spots: Extracted neighborhood voxels from the raw movie containing
        the spots for a trace, with timepoints indexed over the 0-th axis. Each timepoint
        is extracted from a `transcription_pipeline.spot_pipeline.Spot.spot_dataframe`
        object.
    :type trace_raw_spots: np.ndarray
    :param trace_t_s: Time in seconds corresponding to each element of the trace (0-th axis of
        `trace_raw_spots`).
    :type trace_t_s: np.ndarray
    :param trace_centroids: Estimated centroid of spot in each timepoint of `trace_raw_spots`.
    :type trace_centroids: np.ndarray
    :param float window: Window size in seconds over which to average the background.
    :param win_type: Type of weighing to give the timepoints in each window when calculating
         the rolling average. `boxcar` is unweighted average and is much faster, but has
         worse frequency-space performance than `gaussian` (this might matter for e.g.
         autocorrelation-type analysis).
    :type win_type: {"gaussian", "boxcar"}
    :param float std: Standard deviation of the gaussian weighing to use when performing the rolling
        average with `win_type="gaussian"`.
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the  xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :param background: Choose whether the background returned is the mean background
        intensity per pixel or the total background subtracted over the spot.
    :type background: {"mean", "total"}
    :return: Tuple of trace intensity after background averaging, standard error in
        trace intensity, estimated background, and standard error in estimated background.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    trace_intensity = np.zeros_like(trace_t_s, dtype=float)
    trace_intensity_error = np.zeros_like(trace_t_s, dtype=float)
    trace_background = np.zeros_like(trace_t_s, dtype=float)
    trace_background_err = np.zeros_like(trace_t_s, dtype=float)

    for i in range(trace_t_s.size):
        t_s = trace_t_s[i]
        background_pixels, background_pixels_weights = _collect_background_pixels(
            trace_raw_spots,
            t_s,
            trace_t_s,
            trace_centroids,
            window=window,
            win_type=win_type,
            std=std,
            mppZ=mppZ,
            mppYX=mppYX,
            ball_diameter_um=ball_diameter_um,
            shell_width_um=shell_width_um,
            aspect_ratio=aspect_ratio,
        )

        intensity, intensity_error, background_intensity, background_intensity_error = (
            fitting.bootstrap_intensity(
                trace_raw_spots[i],
                centroid=trace_centroids[i],
                mppZ=mppZ,
                mppYX=mppYX,
                ball_diameter_um=ball_diameter_um,
                shell_width_um=shell_width_um,
                aspect_ratio=aspect_ratio,
                num_bootstraps=num_bootstraps,
                background=background,
                background_pixels=background_pixels,
                background_pixels_weights=background_pixels_weights,
            )
        )

        trace_intensity[i] = intensity
        trace_intensity_error[i] = intensity_error
        trace_background[i] = background_intensity
        trace_background_err[i] = background_intensity_error

    return (
        trace_intensity,
        trace_intensity_error,
        trace_background,
        trace_background_err,
    )


def _refine_traces_dataframe(
    compiled_dataframe,
    *,
    window,
    win_type,
    std,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
    num_bootstraps,
    background,
):
    """
    Helper function to average background over all compiled traces in a dataframe.
    """

    def _refine_trace_func(row):
        trace_raw_spots = row["raw_spot"]
        trace_t_s = row["t_s"]
        trace_centroids = row["centroid"]

        refined_trace = refine_trace(
            trace_raw_spots,
            trace_t_s,
            trace_centroids,
            window=window,
            win_type=win_type,
            std=std,
            mppZ=mppZ,
            mppYX=mppYX,
            ball_diameter_um=ball_diameter_um,
            shell_width_um=shell_width_um,
            aspect_ratio=aspect_ratio,
            num_bootstraps=num_bootstraps,
            background=background,
        )

        return refined_trace

    compiled_df = compiled_dataframe.copy()
    compiled_df[
        [
            "refined_intensity_from_neighborhood",
            "refined_intensity_std_error_from_neighborhood",
            "refined_trace_background",
            "refined_trace_background_std_error",
        ]
    ] = compiled_df.apply(_refine_trace_func, axis=1, result_type="expand")

    return compiled_df


def refine_compile_traces(
    tracked_spots_dataframe,
    *,
    min_frames,
    pos_columns=["z", "y", "x"],
    compile_columns=[],
    nuclear_tracking_dataframe=None,
    compile_columns_nuclear=["nuclear_cycle"],
    ignore_negative_spots=True,
    background_averaging=False,
    window=None,
    win_type=None,
    std=None,
    mppZ=None,
    mppYX=None,
    ball_diameter_um=None,
    shell_width_um=None,
    aspect_ratio=None,
    num_bootstraps=None,
    background=None,
    client=None,
    partitions_per_worker=10,
):
    """
    Improves the quantification of a trace by averaging over a few frames to estimate
    the spot background, while still estimating spot intensity on a frame-by-frame basis.
    This should only be used for traces with sampling times much shorter than timescales
    over which the background varies significantly (~30s in preliminary tests).

    :param tracked_spots_dataframe: DataFrame containing information about detected,
        filtered and tracked spots.
    :type tracked_spots_dataframe: pandas DataFrame
    :param int min_frames: Minimum number of frames in a trace for it to be included
        in the compiled traces.
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param compile_columns: List of properties to extract and compile from
        `spot_tracking_dataframe`.
    :type compile_columns: List of column names. Entries can be strings pointing
        to column names, or single-entry dictionaries with the key pointing to the
        column name to compile from, and the value pointing to the new column name
        to give the compiled property in the compiled dictionary.
    :param nuclear_tracking_dataframe: DataFrame containing information about detected
        and tracked nuclei.
    :type nuclear_tracking_dataframe: pandas DataFrame
    :param compile_columns_nuclear: List of properties to extract and compile from
        `nuclear_tracking_dataframe`.
    :type compile_columns_nuclear: List of column names.
    :param bool ignore_negative_spots: Ignores datapoints where the spot quantification
        goes negative - as long as we are looking at background-subtracted intensity,
        negative values are clear mistrackings/misquantifications.
    :param bool background_averaging: If `True`, the background of each trace is averaged
        (and bootstrapped) over multiple frames to improve the intensity quantification.
        Otherwise, traces are compiled as-is.
    :param float window: Window size in seconds over which to average the background.
    :param win_type: Type of weighing to give the timepoints in each window when calculating
         the rolling average. `boxcar` is unweighted average and is much faster, but has
         worse frequency-space performance than `gaussian` (this might matter for e.g.
         autocorrelation-type analysis).
    :type win_type: {"gaussian", "boxcar"}
    :param float std: Standard deviation of the gaussian weighing to use when performing the rolling
        average with `win_type="gaussian"`.
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the  xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :param background: Choose whether the background returned is the mean background
        intensity per pixel or the total background subtracted over the spot.
    :type background: {"mean", "total"}
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param int partitions_per_worker: Number of partitions to split the dataframe containing
        compiled traces into before round-robin sending to workers in the Dask client for
        background averaging and refinement of the quantification. More partitioning adds overhead
        in moving data around, but make sure the workers spend less time idle once the processing
        is done on one of the partitions.
    :return: DataFrame of compiled data indexed by particle, with the intensity timeseries
        refined by averaging and bootstrapping over a few frames if `background_averaging=True`.
    :rtype: pd.DataFrame
    """
    tqdm.write("Compiling traces.")
    compiled_dataframe = _compile_traces_background_averaging(
        tracked_spots_dataframe,
        min_frames=min_frames,
        pos_columns=pos_columns,
        compile_columns=compile_columns,
        nuclear_tracking_dataframe=nuclear_tracking_dataframe,
        compile_columns_nuclear=compile_columns_nuclear,
        ignore_negative_spots=ignore_negative_spots,
    )

    if background_averaging:
        tqdm.write("Refining traces by averaging background.")
        if client is None:
            refined_compiled_df = _refine_traces_dataframe(
                compiled_dataframe,
                window=window,
                win_type=win_type,
                std=std,
                mppZ=mppZ,
                mppYX=mppYX,
                ball_diameter_um=ball_diameter_um,
                shell_width_um=shell_width_um,
                aspect_ratio=aspect_ratio,
                num_bootstraps=num_bootstraps,
                background=background,
            )
        else:
            # Add empty columns to dataframe to make metadata handling easier
            compiled_dataframe[
                [
                    "refined_intensity_from_neighborhood",
                    "refined_intensity_std_error_from_neighborhood",
                    "refined_trace_background",
                    "refined_trace_background_std_error",
                ]
            ] = None

            compiled_dataframe[
                [
                    "refined_intensity_from_neighborhood",
                    "refined_intensity_std_error_from_neighborhood",
                    "refined_trace_background",
                    "refined_trace_background_std_error",
                ]
            ] = compiled_dataframe[
                [
                    "refined_intensity_from_neighborhood",
                    "refined_intensity_std_error_from_neighborhood",
                    "refined_trace_background",
                    "refined_trace_background_std_error",
                ]
            ].astype(
                object
            )

            # Convert compiled dataframe to dask with one partition per
            # worker
            num_workers = len(client.scheduler_info()["workers"])
            refined_compiled_dd = dd.from_pandas(
                compiled_dataframe, npartitions=num_workers * partitions_per_worker
            )

            refined_compiled_df = client.compute(
                refined_compiled_dd.map_partitions(
                    _refine_traces_dataframe,
                    window=window,
                    win_type=win_type,
                    std=std,
                    mppZ=mppZ,
                    mppYX=mppYX,
                    ball_diameter_um=ball_diameter_um,
                    shell_width_um=shell_width_um,
                    aspect_ratio=aspect_ratio,
                    num_bootstraps=num_bootstraps,
                    background=background,
                    meta=refined_compiled_dd,
                )
            ).result()

    else:
        refined_compiled_df = compiled_dataframe

    return refined_compiled_df
