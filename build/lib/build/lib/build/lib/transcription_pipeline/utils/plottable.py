def _generate_trace_plot(particle_index, compiled_dataframe):
    """
    Generates a tuple of the time and intensity vectors for a given particle, and the
    particle ID and assigned nuclear cycle of the trace.
    """
    time = (
        compiled_dataframe.loc[particle_index, "t_s"]
        - compiled_dataframe.at[particle_index, "division_time"]
    )
    intensity = compiled_dataframe.loc[particle_index, "intensity_from_fit"]

    nc = compiled_dataframe.at[particle_index, "nuclear_cycle"]
    particle = compiled_dataframe.at[particle_index, "particle"]

    return (time, intensity, particle, nc)


def generate_trace_plot_list(compiled_dataframe):
    """
    Generates list of tuples of the time and intensity vectors for a given particle,
    and the particle ID and assigned nuclear cycle of the trace.

    :param compiled_dataframe: DataFrame of compiled data indexed by particle.
    :type compiled_dataframe: pandas DataFrame
    :return: List of tuples of the form `(time, intensity, particle, nc)` where `time`
        and `intensity` are vectors corresponding to the trace for `particle`.
    :rtype: List of tuples
    """
    particle_indices = compiled_dataframe.index.values
    trace_list = []
    for particle in particle_indices:
        trace_list.append(_generate_trace_plot(particle, compiled_dataframe))

    return trace_list