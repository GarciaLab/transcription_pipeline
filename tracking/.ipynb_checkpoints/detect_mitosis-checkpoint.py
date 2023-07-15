import numpy as np


def tracks_start_end(tracked_dataframe, min_track_length=1):
    """
    Uses input `tracked_dataframe` with tracking information to construct
    sub-dataframes with entries corresponing to the start and end of a particle track
    (i.e. the first and last frames it is present in respectively) and a sub-dataframe
    of all singletons (particles not connected in any other frames).
    :param tracked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
    :param int min_track_length: Minimum number of frames that a new track must
        span to be considered. This helps reduce spurious detections.
    :return: Tuple(`track_first_frames`, `track_last_frames`, `track_singletons`) where
        *`track_first_frames` contains all rows in the input `tracked_dataframe`
        corresponding to the start of a particle track.
        *`track_last_frames` contains all rows in the input `tracked_dataframe`
        corresponding to the end of a particle track.
        *`track_singletons` contains all rows in the input `tracked_dataframe`
        corresponding to disconnected particles.
    :rtype: Tuple of pandas DataFrames
    """
    first_frame = []
    last_frame = []
    short_tracks = []

    for particle_group in tracked_dataframe.groupby("particle"):
        _, particle = particle_group
        if particle.shape[0] <= min_track_length:
            short_tracks.append(particle.index[0])
        else:
            first_frame.append(particle["frame"].idxmin())
            last_frame.append(particle["frame"].idxmax())

    track_first_frames = tracked_dataframe.iloc[first_frame]
    track_last_frames = tracked_dataframe.iloc[last_frame]
    track_short_tracks = tracked_dataframe.iloc[short_tracks]

    return track_first_frames, track_last_frames, track_short_tracks


def _find_sibling(
    tracked_dataframe,
    track_start_row,
    pos_columns,
    search_range_mitosis,
    antiparallel_threshold,
):
    """
    Finds the index in `tracked_dataframe` of the likeliest sibling of a new particle
    (given by the row `track_start_row` of `tracked_dataframe`). This determines
    candidate siblings based on proximity (within a cuboid of dimensions set by
    `search_range_mitosis` of the particle centroid) and antiparallel velocity vectors
    as determined by a thresholded normalized dot product (an `antiparallel_threshold`
    value of 0 corresponds to perfectly antiparallel vectors, 1 corresponds to
    orthogonal vectors). Within any remaining candidates, the nearest-neighbor is
    returned.
    """
    # Select subdataframe for first frame of this particle
    frame = track_start_row["frame"]
    position = np.array([track_start_row[pos] for pos in pos_columns])
    vel_columns = ["".join(["v_", pos]) for pos in pos_columns]
    frame_subdataframe = tracked_dataframe[tracked_dataframe["frame"] == frame]

    # Select for points within some search cuboid of the new particle. Fall back
    # on cubic search range if `search_range_mitosis` given as a scalar
    for i, pos in enumerate(pos_columns):
        try:
            search_range = search_range_mitosis[i]
        except TypeError:
            search_range = search_range_mitosis

        frame_subdataframe = frame_subdataframe[
            (frame_subdataframe[pos] - position[i]).abs() < search_range
        ]

    # Find positions of midpoints of candidates and particle
    candidate_positions = frame_subdataframe[pos_columns]
    candidate_midpoints = (candidate_positions + position) / 2

    # Find velocity of new track relative to candidate midpoint
    new_track_vel = position - candidate_midpoints
    direction_vector = new_track_vel.divide(
        new_track_vel.apply(np.linalg.norm, axis=1), axis=0
    )

    # Select for tracks with antiparallel velocities within some threshold
    candidate_velocities = frame_subdataframe[vel_columns]
    candidate_direction_vectors = candidate_velocities.divide(
        candidate_velocities.apply(np.linalg.norm, axis=1), axis=0
    )

    dot_product_complement = 1 + (
        candidate_direction_vectors * direction_vector.values
    ).sum(axis=1)

    frame_subdataframe = frame_subdataframe[
        dot_product_complement < antiparallel_threshold
    ]

    # Pick nearest-neighbor of the remaining particles to find the sibling
    if not frame_subdataframe.empty:
        sibling_index = (
            (frame_subdataframe[pos_columns] - position)
            .apply(np.linalg.norm, axis=1)
            .idxmin()
        )
    else:
        sibling_index = None

    return sibling_index


def _assign_siblings(
    tracked_dataframe,
    pos_columns,
    search_range_mitosis,
    antiparallel_threshold,
    min_track_length,
    image_dimensions,
    exclude_border,
):
    """
    Returns an (2, n)-shape ndarray with each element along the 0-th axis corresponding
    respectively to the index in `tracked_dataframe` of a new track and to the index
    of its sibling. This can then be used to construct lineages. Fraction
    `exclude_border` of the image in each border is excluded from the search for
    new particles to assign siblings to.
    """
    if (exclude_border is None) or (image_dimensions is None):
        exclude_border = 0
        image_dimensions = 0

    image_dimensions = np.asarray(image_dimensions)
    try:
        iter(exclude_border)
        exclude_border = np.asarray(exclude_border)
    except TypeError:
        pass

    track_first_frames, _, _ = tracks_start_end(tracked_dataframe, min_track_length)
    track_start_index = []
    siblings = []
    for i, track_start_row in track_first_frames.iterrows():
        coordinates = np.array([track_start_row[pos] for pos in pos_columns])

        start_border = (
            np.abs(coordinates - np.asarray(image_dimensions))
            < exclude_border * image_dimensions
        )
        end_border = np.abs(coordinates) < exclude_border * image_dimensions
        in_border = np.any(np.array([start_border, end_border]))

        if not in_border:
            track_siblings = _find_sibling(
                tracked_dataframe,
                track_start_row,
                pos_columns,
                search_range_mitosis,
                antiparallel_threshold,
            )
        else:
            track_siblings = None

        if track_siblings is not None:
            track_start_index.append(i)
            siblings.append(track_siblings)

    sibling_array = np.array([track_start_index, siblings]).T
    return sibling_array


def _assign_parents(mitosis_dataframe, sibling_array):
    """
    Iterates through a (2, n)-shape array of sibling indices in input
    `mitosis_dataframe` as returned by :func:`~_assign_siblings` to find parents of
    the siblings in the previous frames. Returns indices of the parents as an ndarray
    implicitly positionally indexed as per `sibling_array`, with `np.nan` values
    when a parent cannot be found.
    """
    parent_index_array = []
    for sibling_pair in sibling_array:
        parent_particle = mitosis_dataframe["particle"].iloc[sibling_pair[1]]
        parent_subdataframe = mitosis_dataframe["particle"] == parent_particle

        frame = mitosis_dataframe["frame"].iloc[sibling_pair[1]]
        parent_track = parent_subdataframe & (mitosis_dataframe["frame"] < frame)

        parent_frame_series = mitosis_dataframe.loc[parent_track, "frame"]
        if parent_frame_series.empty:
            parent_index = np.nan
        else:
            parent_index = parent_frame_series.idxmax()
        parent_index_array.append(parent_index)

    parent_index_array = np.array(parent_index_array)
    return parent_index_array


def _relabel_sibling(mitosis_dataframe, sibling_array):
    """
    Modifies `particle` labels in `mitosis_dataframe` to assign a new identity to
    the sibling that maintained parent identity after tracking with trackpy.
    """
    new_label = mitosis_dataframe["particle"].max() + 1

    for sibling_pair in sibling_array:
        old_label = mitosis_dataframe["particle"].iloc[sibling_pair[1]]
        sibling_subdataframe = mitosis_dataframe["particle"] == old_label
        division_frame = mitosis_dataframe["frame"].iloc[sibling_pair[0]]
        sibling_new_track = sibling_subdataframe & (
            mitosis_dataframe["frame"] >= division_frame
        )
        mitosis_dataframe.loc[sibling_new_track, "particle"] = new_label
        new_label += 1

    return None


def _add_parent(mitosis_dataframe, sibling_array, parent_index_array):
    """
    Adds a `parent` column to `mitosis_dataframe` and adds the parent identity for
    each identified pair of siblings so that lineage information is preserved.
    """
    mitosis_dataframe["parent"] = np.nan

    for i, sibling_pair in enumerate(sibling_array):
        if not np.isnan(parent_index_array[i]):
            parent_label = mitosis_dataframe["particle"].iloc[
                int(parent_index_array[i])
            ]
            for child in sibling_pair:
                mitosis_dataframe.loc[child, "parent"] = parent_label

    mitosis_dataframe["parent"] = mitosis_dataframe["parent"].astype("Int64")

    return None


def construct_lineage(
    tracked_dataframe,
    *,
    pos_columns,
    search_range_mitosis,
    antiparallel_threshold,
    min_track_length=1,
    image_dimensions=None,
    exclude_border=None,
):
    """
    Constructs lineages in dividing nuclei by iterating over new tracks and identifying
    pairs of siblings after a division, then determining and assigning the parent
    identity for both siblings and clipping the parent track at the division so that
    both siblings have a new identity. Returns a copy of `tracked_dataframe` with
    the identities rearranges so that each nucleus has a unique identity after
    divisions and an added column to keep track of the parent identities.

    :param tracked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type linked_dataframe: pandas DataFrame
    :param pos_columns: Name of columns in segmentation_df containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param search_range_mitosis: Coordinate search region to look for sibling
        candidates, indexed as per `pos_columns`. Assumes isotropic if given as a
        scalar.
    :type search_range_mitosis: scalar or tuple of scalars
    :param antiparallel_threshold: Threshold for (1 + normalized dot product of
        velocity vectors) right after division above which particles are not considered
        to be viable candidate siblings. This is 0 for perfectly antiparallel
        velocities, 1 for orthogonal velocities and 2 for perfectly parallel velocities.
    :type antiparallel_threshold: scalar
    :param int min_track_length: Minimum number of frames that a new track must
        span to be considered. This helps reduce spurious detections.
    :param image_dimensions: Image dimension in each respective coordinate of
        `pos_columns`.
    :type image_dimensions: Array of integers.
    :param exclude_border: Fraction of border in each position coordinate
        to exclude when searching for new particles to assign siblings to.
    :type exclude_border: float or array of floats for each coordinate.
    :return: Copy of `tracked_dataframe` with particle identities corresponding to
        lineage and an added column with parent identities.
    :rtype: pandas DataFrame
    """
    mitosis_dataframe = tracked_dataframe.copy()

    # Construct array of siblings
    sibling_array = _assign_siblings(
        mitosis_dataframe,
        pos_columns,
        search_range_mitosis,
        antiparallel_threshold,
        min_track_length,
        image_dimensions,
        exclude_border,
    )

    # Assign parent by indexing over dataframe
    parent_index_array = _assign_parents(mitosis_dataframe, sibling_array)

    # Assign new labels to siblings and clip parent track
    _relabel_sibling(mitosis_dataframe, sibling_array)

    # Add parents to sibling pairs
    _add_parent(mitosis_dataframe, sibling_array, parent_index_array)

    # Relabel all particles with the label assigned at the first frame
    # by the sibling sorting
    for _, track in mitosis_dataframe.groupby("particle"):
        particle = track["particle"].values[0]
        orphan = track["parent"].notnull().sum() == 0
        if not orphan:
            mitosis_dataframe.loc[
                mitosis_dataframe["particle"] == particle, "parent"
            ] = track["parent"][track["parent"].notnull()].values[0]

    return mitosis_dataframe


def tracks_to_napari(viewer, dataframe, name='tracks', output=False):
    """
    Uses tracking and lineage information in `mitosis_dataframe` to construct
    tracks datastructures compatible with napari tracks layer. If output is requested,
    will also return these data structures.

    :param viewer: napari viewer.
    :type viewer: napari viewer object
    :param dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe` and lineage construction with `construct_lineage`.
    :type dataframe: pandas DataFrame
    :param bool output: If True, returns the napari-compatible datastructures used
        to visualize tracks.
    :return: If `output = True`, returns lineage dictionary and parent properties
        used to visualize tracks in napari.
    :rtype: {Tuple, None}
    """
    mitosis_dataframe = dataframe.copy()
    mitosis_dataframe['frame'] = mitosis_dataframe['frame'] - 1
    
    tracks = mitosis_dataframe.loc[:, ["particle", "frame", "z", "y", "x"]].to_numpy()
    
    lineage = {}
    for _, family in mitosis_dataframe.groupby("parent"):
        parent = (family["parent"].values)[0]
        children = family["particle"].to_list()
        lineage[parent] = children

    properties = {
        "parent": [
            mitosis_dataframe.loc[
                mitosis_dataframe["particle"] == idx, "parent"
            ].values.fillna(0)[0]
            for idx in tracks[:, 0]
        ]
    }

    viewer.add_tracks(tracks, properties=properties, graph=lineage, name=name)

    if output:
        out = lineage, properties

    else:
        out = None

    return out