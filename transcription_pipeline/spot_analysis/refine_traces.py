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
