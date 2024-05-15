import pandas as pd
import warnings
import numpy as np
from tqdm import tqdm


def _compile_property(compiled_dataframe_row, original_dataframe, quantity, sort=True):
    """
    Returns the values of a specified column `quantity` in `original_dataframe`
    corresponding to a particle specified by the `particle` column of
    `compiled_dataframe_row`.
    """
    particle_df = original_dataframe[
        original_dataframe["particle"] == compiled_dataframe_row["particle"]
    ]

    if sort:
        particle_df = particle_df.sort_values("t_s")

    compiled_property = particle_df[quantity].values

    return compiled_property


def compile_traces(
    spot_tracking_dataframe,
    compile_columns_spot=[
        "frame",
        "t_s",
        "intensity_from_neighborhood",
        "intensity_std_error_from_neighborhood",
    ],
    nuclear_tracking_dataframe=None,
    compile_columns_nuclear=["nuclear_cycle"],
    max_frames_outside_division=4,
    ignore_negative_spots=True,
):
    """
    Compiles spot tracking data (and nuclear tracking data if provided) by particles,
    with particles indexed by row and properties of interest indexed by column such that
    the cell corresponding to a (particle, property) pair contains an array tracking
    the value of that property across time.

    :param spot_tracking_dataframe: DataFrame containing information about detected,
        filtered and tracked spots.
    :type spot_tracking_dataframe: pandas DataFrame
    :param compile_columns_spot: List of properties to extract and compile from
        `spot_tracking_dataframe`.
    :type compile_columns_spot: List of column names. Entries can be strings pointing
        to column names, or single-entry dictionaries with the key pointing to the
        column name to compile from, and the value pointing to the new column name
        to give the compiled property in the compiled dictionary.
    :param nuclear_tracking_dataframe: DataFrame containing information about detected
        and tracked nuclei.
    :type nuclear_tracking_dataframe: pandas DataFrame
    :param compile_columns_nuclear: List of properties to extract and compile from
        `nuclear_tracking_dataframe`.
    :type compile_columns_nuclear: List of column names.
    :param int max_frames_outside_division: The maximum number of timepoints a track
        can have outside of a nuclear cycle and still be considered exclusively part
        of that nuclear cycle.
    :param bool ignore_negative_spots: Ignores datapoints where the spot quantification
        goes negative - as long as we are looking at background-subtracted intensity,
        negative values are clear mistrackings/misquantifications.
    :return: DataFrame of compiled data indexed by particle.
    :rtype: pd.DataFrame
    """
    if ignore_negative_spots:
        try:
            spot_tracking_dataframe = spot_tracking_dataframe[
                spot_tracking_dataframe["intensity_from_neighborhood"] > 0
            ].copy()
        except KeyError:
            pass

    particles = np.sort(np.trim_zeros(spot_tracking_dataframe["particle"].unique()))

    compiled_dataframe = pd.DataFrame(data=particles, columns=["particle"])

    for quantity in compile_columns_spot:
        # Preallocate and convert to dtype object to allow storage of arrays
        compiled_dataframe[quantity] = np.nan
        compiled_dataframe[quantity] = compiled_dataframe[quantity].astype(object)

        try:
            compiled_dataframe[quantity] = compiled_dataframe.apply(
                _compile_property, args=(spot_tracking_dataframe, quantity), axis=1
            )
        except KeyError:
            warnings.warn(
                "".join(
                    [
                        "Property ",
                        quantity,
                        " to compile not found, check",
                        " that names of provided columns match respective DataFrame.",
                    ]
                ),
                stacklevel=2,
            )

    if nuclear_tracking_dataframe is not None:
        for quantity in compile_columns_nuclear:
            if isinstance(quantity, dict):
                property_key = list(quantity.keys())[0]
                property_value = quantity[property_key]
                quantity = property_key
                column_name = property_value
            else:
                column_name = quantity

            if quantity != "division_time":
                # Preallocate and convert to dtype object to allow storage of arrays
                compiled_dataframe[column_name] = np.nan
                compiled_dataframe[column_name] = compiled_dataframe[
                    column_name
                ].astype(object)

                compiled_dataframe[column_name] = compiled_dataframe.apply(
                    _compile_property,
                    args=(nuclear_tracking_dataframe, quantity),
                    axis=1,
                )

        if "division_time" in compile_columns_nuclear:
            compiled_dataframe["division_time"] = compiled_dataframe.apply(
                lambda x: _compile_property(x, nuclear_tracking_dataframe, "t_s").min(),
                axis=1,
            )

    # If nuclear cycle is available, collapse array so we get a single value in the cell
    # If there is a mitosis detection failure and traces are joined such that they lie
    # in multiple nuclear cycles, the array of nuclear cycle values corresponding to each
    # timepoint is saved instead.
    if "nuclear_cycle" in compiled_dataframe:

        def _assign_nuclear_cycle(compiled_dataframe_row):
            nuclear_cycle_timepoints = compiled_dataframe_row["nuclear_cycle"]
            included_nuclear_cycles, num_frames_cycle = np.unique(
                nuclear_cycle_timepoints, return_counts=True
            )
            nc = included_nuclear_cycles[num_frames_cycle > max_frames_outside_division]
            if nc.size == 1:
                nc = nc[0]
            return nc

        compiled_dataframe["nuclear_cycle"] = compiled_dataframe.apply(
            _assign_nuclear_cycle,
            axis=1,
        )

    return compiled_dataframe


def consolidate_traces(
    traces_dataframe,
    trace_column="background_intensity_from_neighborhood",
    time_column="t_frame",
):
    """
    Consolidates all traces from a compiled dataframe structure as output by
    :func:`~compile_traces` into a single array with dimensions
    `(number of traces, number of time points in the longest trace)`. All missing time points
    are padded with `np.nan`.

    :param traces_dataframe: Dataframe containing compiled traces.
    :type traces_dataframe: pandas.DataFrame
    :param str trace_column: Name of column in `traces_dataframe` containing the
        traces to be consolidated (each entry being a time series array, each row
        corresponding to a single trace).
    :param str time_column: Name of column in `traces_dataframe` containing the
        time points of the traces to be consolidated in units of the time resolution.
        This must be an array of integers that map to real time (not just frame number
        since those can be different if the data is concatenated from multiple series
        with a time delay between the end of a series and the start of the next).
    :return: Padded array containing compiled traces.
    :rtype: np.ndarray
    """
    # Initialize as pandas object for convenient use of `apply`.
    traces_dataframe = traces_dataframe.copy().reset_index(inplace=False, drop=True)
    traces_time_series = traces_dataframe[time_column]

    traces_length = traces_time_series.apply(lambda x: x[-1] - x[0] + 1)
    num_traces = traces_time_series.size
    consolidated_trace_array = np.empty((num_traces, traces_length.max()), dtype=float)
    consolidated_trace_array[:] = np.nan

    def _inject_trace(row):
        """
        Helper function to inject traces into consolidated numpy array, padding
        missing timepoints with `NaN`s.
        """
        trace_index = row[time_column] - row[time_column][0]
        consolidated_trace_array[row.name][trace_index] = row[trace_column]

    tqdm.pandas(desc="Padding and consolidating traces")
    traces_dataframe.progress_apply(_inject_trace, axis=1)

    return consolidated_trace_array
