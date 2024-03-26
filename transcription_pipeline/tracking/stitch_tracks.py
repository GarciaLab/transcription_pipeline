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
    tracked_dataframe,
    pos_columns,
    max_distance,
    max_frame_distance,
    frames_mean=4,
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
    :param int frames_mean: Number of frames to average over when estimating the mean
        position of the start and end of candidate tracks to stitch together.
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
    # Only consider filtered particles
    filtered_mask = tracked_dataframe["particle"] != 0
    filtered_dataframe = tracked_dataframe[filtered_mask].copy()

    # We keep stitching information in a pandas DataFrame
    stitch_dataframe = pd.DataFrame()

    # Find all unique particles
    particles = np.sort(np.trim_zeros(filtered_dataframe["particle"].unique()))
    stitch_dataframe["preliminary_particles"] = particles

    # Calculate the mean position for start and end of each trace.
    def _mean_trace_position(particle_row, pos, frames_mean, section):
        """
        Calculates mean position over `frames_mean` frames at the start or
        end of a track depending on whether `section` is set to "start" or
        "end" respectively.
        """
        particle_sub_df = filtered_dataframe[
            filtered_dataframe["particle"] == particle_row["preliminary_particles"]
        ].sort_values("t_s")
        track_pos = particle_sub_df[pos]

        if track_pos.size > frames_mean:
            mean_pos = track_pos.mean()
        else:
            if section == "end":
                mean_pos = track_pos[-frames_mean:].mean()
            elif section == "start":
                mean_pos = track_pos[:frames_mean].mean()
            else:
                raise ValueError("Could not recognize `section` parameter.")

        return mean_pos

    for pos in pos_columns:
        column = "".join(["start_", pos])
        stitch_dataframe[column] = stitch_dataframe.apply(
            _mean_trace_position, args=(pos, frames_mean, "start"), axis=1
        )

        column = "".join(["end_", pos])
        stitch_dataframe[column] = stitch_dataframe.apply(
            _mean_trace_position, args=(pos, frames_mean, "end"), axis=1
        )

    # Pull first and last frame of each particle
    def _first_last_frames(particle_row):
        """Pulls the first and last frame."""
        particle_sub_df = filtered_dataframe[
            filtered_dataframe["particle"] == particle_row["preliminary_particles"]
        ]
        first_frame = particle_sub_df["frame"].min()
        last_frame = particle_sub_df["frame"].max()

        return first_frame, last_frame

    stitch_dataframe[["first_frame", "last_frame"]] = stitch_dataframe.apply(
        _first_last_frames, axis=1, result_type="expand"
    )

    # Find the nearest-neighbor that starts within a few frames of the
    # current particle.
    def _nearest_neighbor_iter(particle, max_frame_distance_iter):
        """
        Finds and computes the distance to the nearest neighbor that starts at most
        `max_frame_distance_iter` from the end of the current particle.
        """
        # Pull end position of current track
        end_pos_columns = ["".join(["end_", pos]) for pos in pos_columns]
        particle_pos = stitch_dataframe.loc[
            stitch_dataframe["preliminary_particles"] == particle, end_pos_columns
        ]

        # Look at distance of start point of other tracks from end point of
        # current track.
        start_pos_columns = ["".join(["start_", pos]) for pos in pos_columns]
        displacement = stitch_dataframe[start_pos_columns] - particle_pos.values

        distance = ((displacement**2).sum(axis=1)) ** 0.5

        # Mask all tracks that have consistent start times with the end of
        # the current track.
        start_end_difference = (
            stitch_dataframe["first_frame"]
            - stitch_dataframe.loc[
                stitch_dataframe["preliminary_particles"] == particle, "last_frame"
            ].values
        )
        start_end_mask = (start_end_difference >= 0) & (
            start_end_difference <= max_frame_distance_iter
        )

        distance_df = distance[
            (stitch_dataframe["preliminary_particles"] != particle) & start_end_mask
        ]
        if distance_df.size > 0:
            nearest_neighbor_idx = distance_df.idxmin()

            nearest_neighbor = stitch_dataframe["preliminary_particles"].iloc[
                nearest_neighbor_idx
            ]
            nearest_neighbor_distance = distance.iloc[nearest_neighbor_idx]

        else:
            nearest_neighbor = 0
            nearest_neighbor_distance = np.inf

        return nearest_neighbor, nearest_neighbor_distance

    def _nearest_neighbor(particle_row):
        """
        Finds and computes the distance to the closest particle in time that is within
        some radius of the current particle.
        """
        particle = particle_row["preliminary_particles"]

        nearest_neighbor = 0
        nearest_neighbor_distance = np.inf

        for frame_distance in range(max_frame_distance):
            nearest_neighbor, nearest_neighbor_distance = _nearest_neighbor_iter(
                particle, frame_distance
            )
            if nearest_neighbor_distance < max_distance:
                break

        return nearest_neighbor, nearest_neighbor_distance

    stitch_dataframe[["nearest_neighbor", "distance"]] = stitch_dataframe.apply(
        _nearest_neighbor, axis=1, result_type="expand"
    )

    stitch_dataframe["nearest_neighbor"] = stitch_dataframe["nearest_neighbor"].astype(
        int
    )

    link_mask = stitch_dataframe["distance"] < max_distance

    stitch_dataframe["link"] = link_mask

    return stitch_dataframe


def stitch_tracks(
    tracked_dataframe,
    pos_columns,
    max_distance,
    max_frame_distance,
    frames_mean=4,
    quantification="intensity_from_neighborhood",
    inplace=False,
):
    """
    Stitches together tracks depending on proximity in time (start time of candidate
    track to stich relative to end time of a given track) and space (position averaged
    over a few frames at the end of given track relative to start of candidate track).

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
    :param int frames_mean: Number of frames to average over when estimating the mean
        position of the start and end of candidate tracks to stitch together.
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
        tracked_df, pos_columns, max_distance, max_frame_distance, frames_mean
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

    linking_sub_df.apply(_stitch_particle_track, axis=1)

    remove_duplicates(tracked_df, quantification=quantification)

    if inplace:
        return None
    else:
        return tracked_df