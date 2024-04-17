import pandas as pd
import warnings
import numpy as np


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
    :rtype: pandas DataFrame
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
