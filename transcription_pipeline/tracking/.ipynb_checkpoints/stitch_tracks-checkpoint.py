import pandas as pd
import numpy as np

# We can manually stitch together traces that have been neglected by
# tracking - this can be necessary when the spot detection is noisy
# and we can use more global time-domain information about persistence
# and intersection of preliminary traces that Crocker-Grier doesn't
# necessarily care about. There's probably a way to use mean position,
# mean time, and persistence of traces as "coordinates" and reformulate
# it in a way that is tractable by Crocker-Grier, but this seems overkill
# for now.


def remove_duplicates(tracked_dataframe, quantification="intensity_from_neighborhood"):
    """
    After a round of tracking, a stitching operation is usually carried out to link
    partial tracks that are easily resolvable by mean position, leaving some
    tracks with duplicate spots in the frame where the stitching occurs. This function
    scans through the tracked and stitched dataframe to remove duplicate spots.
    Modifies the dataframe in place.

    :param tracked_dataframe: DataFrame containing information about detected,
        filtered and tracked spots after stitching operation.
    :type tracked_dataframe: pandas DataFrame
    :param str quantification: Name of dataframe column containing quantification to use
        when compiling successive differences along a trace. Defaults to
        `intensity_from_neighborhood`.
    :return: None
    """
    filtered_mask = tracked_dataframe["particle"] != 0
    filtered_dataframe = tracked_dataframe[filtered_mask]

    def _remove_duplicate_row(filtered_dataframe_row):
        """
        Checks every row of `filtered_dataframe` for spots with the same
        label in the same frame, and chooses the brighter of the two.
        """
        particle = filtered_dataframe_row["particle"]
        frame = filtered_dataframe_row["frame"]
        particle_frame_mask = (tracked_dataframe["particle"] == particle) & (
            tracked_dataframe["frame"] == frame
        )

        if particle_frame_mask.sum() > 1:
            particle_frame_subdf = tracked_dataframe[particle_frame_mask]
            brightest_spot_mask = (
                tracked_dataframe[quantification]
                == particle_frame_subdf[quantification].max()
            ) & particle_frame_mask
            tracked_dataframe[(~brightest_spot_mask) & particle_frame_mask] = 0

    filtered_dataframe.apply(_remove_duplicate_row, axis=1)


def construct_stitch_dataframe(
    tracked_dataframe, pos_columns, max_distance, max_frame_distance
):
    """
    :param tracked_dataframe: DataFrame containing information about detected,
        filtered and tracked spots.
    :type tracked_dataframe: pandas DataFrame
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param float max_distance: Maximum distance between mean position of partial tracks
        that still allows for stitching to occur.
    :param int max_frame_distance: Maximum number of frames between tracks with no
        points from either tracks that still allows for stitching to occur.
    :return: DataFrame with columns {`preliminary_particles`, `nearest_neighbor`,
        `distance`, `frame_overlap`, `frame_distance`}.
        *`preliminary_particles`: Particle ID of tracked particles in `tracked_dataframe`.
        *`nearest_neighbor`: Mean-position nearest-neighbor of current particle.
        *`distance`: Euclidean distance (with respect to coordinates specified in
        `pos_columns`) to nearest neighbor.
        *`frame_overlap`: Number of frames that include both tracks.
        *`frame_distance`: Number of frames separating the end of the earlier track
        and the beginning of the later track if there is no overlapping frame.
    :rtype: pandas DataFrame
    """
    # We keep stitching information in a pandas DataFrame
    stitch_dataframe = pd.DataFrame()

    # Find all unique particles
    particles = np.sort(np.trim_zeros(tracked_dataframe["particle"].unique()))
    stitch_dataframe["preliminary_particles"] = particles

    # Calculate the mean position for each trace across all timepoints.
    def _mean_trace_position(particle_row, pos):
        """
        Calculates mean position across all timepoints in a trace for a given
        particle.
        """
        particle_sub_df = tracked_dataframe[
            tracked_dataframe["particle"] == particle_row["preliminary_particles"]
        ]
        return particle_sub_df[pos].mean()

    for pos in pos_columns:
        column = "".join(["mean_", pos])
        stitch_dataframe[column] = stitch_dataframe.apply(
            _mean_trace_position, args=(pos,), axis=1
        )

    # Now we can start to decide whether to link traces or not. We can set this
    # up as a simple series of steps:
    ##  1. Find the nearest neighbor of a particle, and check whether it is
    ## within an appropriate radius is mean position.
    ## 2. If within an appropriate radius, check the overlapping number of frames.
    ## We expect this to be at most 1 for mistrackings due to spurious particle
    ## assignments.
    ## 3. If overlap is 1, we are done. If the overlap is >1, the linking is
    ## rejected. If there is no overlap, check whether the distance in time
    ## is acceptable (earlier traces ends within a few frames of start of first
    ## frame).

    def _nearest_neighbor(particle):
        """
        Finds and computes the distance to the nearest neighbor.
        """
        mean_pos_columns = ["".join(["mean_", pos]) for pos in pos_columns]
        particle_pos = stitch_dataframe.loc[
            stitch_dataframe["preliminary_particles"] == particle, mean_pos_columns
        ]

        displacement = stitch_dataframe[mean_pos_columns] - particle_pos.values

        distance = ((displacement**2).sum(axis=1)) ** 0.5

        nearest_neighbor_idx = distance[
            stitch_dataframe["preliminary_particles"] != particle
        ].idxmin()
        nearest_neighbor = stitch_dataframe["preliminary_particles"].iloc[
            nearest_neighbor_idx
        ]
        nearest_neighbor_distance = distance.iloc[nearest_neighbor_idx]
        return nearest_neighbor, nearest_neighbor_distance

    def _frame_overlap_nearest_neighbor(particle_row):
        """
        Computes the number of overlapping frames with the mean-position
        nearest-neighbor. If there is no overlap, computes the number
        of frames separating the two traces (defined as the difference
        between the frame in which the earlier trace disappears and the
        frame in which the later trace first appears).
        """
        particle = particle_row["preliminary_particles"]

        nearest_neighbor, nearest_neighbor_distance = _nearest_neighbor(particle)
        time_vector_particle = tracked_dataframe.loc[
            tracked_dataframe["particle"] == particle, "frame"
        ].values
        time_vector_nn = tracked_dataframe.loc[
            tracked_dataframe["particle"] == nearest_neighbor, "frame"
        ].values

        frame_overlap = np.isin(time_vector_particle, time_vector_nn).sum()

        if frame_overlap == 0:
            if time_vector_particle.min() < time_vector_nn.min():
                frame_distance = time_vector_nn.min() - time_vector_particle.max()
            else:
                frame_distance = time_vector_particle.min() - time_vector_nn.max()

        else:
            frame_distance = 0

        return (
            nearest_neighbor,
            nearest_neighbor_distance,
            frame_overlap,
            frame_distance,
        )

    stitch_dataframe[
        ["nearest_neighbor", "distance", "frame_overlap", "frame_distance"]
    ] = stitch_dataframe.apply(
        _frame_overlap_nearest_neighbor, axis=1, result_type="expand"
    )

    stitch_dataframe["nearest_neighbor"] = stitch_dataframe["nearest_neighbor"].astype(
        int
    )

    # We can now decide whether to link traces to their nearest neighbor.
    distance, frame_overlap, frame_distance = stitch_dataframe[
        ["distance", "frame_overlap", "frame_distance"]
    ].T.to_numpy()

    link_mask = (distance < max_distance) & (
        (frame_overlap == 1)
        | ((frame_overlap == 0) & (frame_distance < max_frame_distance))
    )

    stitch_dataframe["link"] = link_mask

    return stitch_dataframe


def stitch_tracks(
    tracked_dataframe,
    pos_columns,
    max_distance,
    max_frame_distance,
    quantification="intensity_from_neighborhood",
    inplace=False,
):
    """
    :param tracked_dataframe: DataFrame containing information about detected,
        filtered and tracked spots.
    :type tracked_dataframe: pandas DataFrame
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param float max_distance: Maximum distance between mean position of partial tracks
        that still allows for stitching to occur.
    :param int max_frame_distance: Maximum number of frames between tracks with no
        points from either tracks that still allows for stitching to occur.
    :param str quantification: Name of dataframe column containing quantification to use
        when compiling successive differences along a trace. Defaults to
        `intensity_from_neighborhood`.
    :param bool inplace: If `True`, `tracked_dataframe` is modified in-place and the
        function returns `None`. Otherwise, a stitched copy is returned.
    :return: Tracked dataframe with stitched tracks.
    :rtype: pandas DataFrame or `None`
    """
    # Stitching is done in a separate copy of the tracked dataframe unless
    # requested.
    if not inplace:
        tracked_df = tracked_dataframe.copy()
    else:
        tracked_df = tracked_dataframe

    # Construct stitching dataframe
    stitch_dataframe = construct_stitch_dataframe(
        tracked_df, pos_columns, max_distance, max_frame_distance
    )

    linking_sub_df = stitch_dataframe[stitch_dataframe["link"]]

    def _stitch_particle_track(linking_sub_df_row):
        """
        Reassigns linked particle with higher particle ID to its nearest
        neighbor.
        """
        particle = linking_sub_df_row["preliminary_particles"]
        linked_neighbor = linking_sub_df_row["nearest_neighbor"]

        if particle == linked_neighbor:
            pass

        elif particle > linked_neighbor:
            tracked_df.loc[
                tracked_df["particle"] == particle, "particle"
            ] = linked_neighbor
            stitch_dataframe.loc[
                stitch_dataframe["preliminary_particles"] == particle,
                "preliminary_particles",
            ] = linked_neighbor
        else:
            tracked_df.loc[
                tracked_df["particle"] == linked_neighbor, "particle"
            ] = particle
            stitch_dataframe.loc[
                stitch_dataframe["preliminary_particles"] == linked_neighbor,
                "preliminary_particles",
            ] = particle

    stitch_dataframe.apply(_stitch_particle_track, axis=1)

    remove_duplicates(tracked_df, quantification=quantification)

    if inplace:
        return None
    else:
        return tracked_df