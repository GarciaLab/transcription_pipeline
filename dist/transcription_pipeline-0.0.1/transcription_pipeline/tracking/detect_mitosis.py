import numpy as np
import trackpy
import deprecation


def tracks_start(
    tracked_dataframe,
    pos_columns,
    min_track_length=1,
    modify_df=True,
    image_dimensions=None,
    exclude_border=None,
    output=False,
):
    """
    Uses input `tracked_dataframe` with tracking information to construct
    sub-dataframes with entries corresponding to the start of a particle track
    (i.e. the first frame it is present in).

    :param tracked_dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe`.
    :type tracked_dataframe: pandas DataFrame
    :param pos_columns: Name of columns in segmentation_df containing a position
        coordinate.
    :param int min_track_length: Minimum number of frames that a new track must
        span to be considered. This helps reduce spurious detections.
    :param bool modify_df: If `True`, the input `tracked_dataframe` is modified
        in-place to add a column of booleans `new_particle` which keeps track of
        whether the particle in every row is a newly-appearing particle.
    :param image_dimensions: Image dimension in each respective coordinate of
        `pos_columns`.
    :type image_dimensions: Array of integers.
    :param exclude_border: Fraction of border in each position coordinate
        to exclude when searching for new particles to assign siblings to.
    :type exclude_border: float or array of floats for each coordinate.
    :param bool output: If `False`, the function returns `None`.
    :return: All rows in the input `tracked_dataframe` corresponding to the start of
        a particle track.
    :rtype: pandas DataFrame
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

    first_frame = []

    for particle_group in tracked_dataframe.groupby("particle"):
        _, particle = particle_group
        initial_frame_index = particle["frame"].idxmin()
        coordinates = np.array(
            [tracked_dataframe[pos].iloc[initial_frame_index] for pos in pos_columns]
        )

        start_border = (
            np.abs(coordinates - np.asarray(image_dimensions))
            < exclude_border * image_dimensions
        )
        end_border = np.abs(coordinates) < exclude_border * image_dimensions
        in_border = np.any(np.array([start_border, end_border]))

        if not (particle.shape[0] <= min_track_length or in_border):
            first_frame.append(initial_frame_index)

    track_first_frames = tracked_dataframe.iloc[first_frame]

    if modify_df:
        new_particles = track_first_frames.index
        tracked_dataframe["new_particle"] = False
        tracked_dataframe.loc[new_particles, "new_particle"] = True

    if not output:
        track_first_frames = None

    return track_first_frames


@deprecation.deprecated(
    details=" ".join(
        [
            "Deprecated in favor of sibling linking using `trackpy` in",
            ":func:`~_assign_siblings_frame` and :func:`~_assign_siblings`, keeping",
            "in the code base for now in case some dataset respond better to",
            "thresholding off of antiparallel velocity vectors.",
        ]
    )
)
def _find_sibling_threshold(
    tracked_dataframe,
    track_start_row,
    pos_columns,
    search_range_mitosis,
    antiparallel_threshold,
):
    """
    Finds the index in `tracked_dataframe` of the
    likeliest sibling of a new particle (given by the row `track_start_row` of
    `tracked_dataframe`).cThis determines candidate siblings based on proximity
    (within a cuboid of dimensions set by `search_range_mitosis` of the particle
    centroid) and antiparallel velocity vectors as determined by a thresholded
    normalized dot product (an `antiparallel_threshold` value of 0 corresponds to
    perfectly antiparallel vectors, 1 corresponds to orthogonal vectors). Within any
    remaining candidates, the nearest-neighbor is returned.
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


@deprecation.deprecated(
    details=" ".join(
        [
            "Deprecated in favor of sibling linking using `trackpy` in",
            ":func:`~_assign_siblings_frame` and :func:`~_assign_siblings`, keeping in"
            "the code base for now in case some dataset respond better to thresholding"
            "off of antiparallel velocity vectors.",
        ]
    )
)
def _assign_siblings_threshold(
    tracked_dataframe,
    pos_columns,
    search_range_mitosis,
    antiparallel_threshold,
    min_track_length,
    image_dimensions,
    exclude_border,
):
    """
    Returns an (2, n)-shape ndarray with each element
    along the 0-th axis corresponding respectively to the index in `tracked_dataframe`
    of a new track and to the index of its sibling. This can then be used to construct
    lineages. Fraction `exclude_border` of the image in each border is excluded from
    the search for new particles to assign siblings to.
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

    track_first_frames = tracks_start(
        tracked_dataframe,
        pos_columns,
        min_track_length=min_track_length,
        modify_df=False,
        image_dimensions=image_dimensions,
        exclude_border=exclude_border,
        output=True,
    )

    track_start_index = []
    siblings = []

    def _iterate_siblings(track_start_row):
        """
        Iterates over row of dataframe of new particles to find siblings.
        """
        coordinates = np.array([track_start_row[pos] for pos in pos_columns])

        start_border = (
            np.abs(coordinates - np.asarray(image_dimensions))
            < exclude_border * image_dimensions
        )
        end_border = np.abs(coordinates) < exclude_border * image_dimensions
        in_border = np.any(np.array([start_border, end_border]))

        if not in_border:
            track_siblings = _find_sibling_threshold(
                tracked_dataframe,
                track_start_row,
                pos_columns,
                search_range_mitosis,
                antiparallel_threshold,
            )
        else:
            track_siblings = None

        if track_siblings is not None:
            track_start_index.append(track_start_row.index)
            siblings.append(track_siblings)

        return None

    track_first_frames.apply(_iterate_siblings)
    sibling_array = np.array([track_start_index, siblings]).T

    return sibling_array


def _assign_siblings_frame(
    frame_df,
    pos_columns,
    search_range,
    antiparallel_coordinate,
    antiparallel_weight,
    minimum_age,
    **kwargs,
):
    """
    Finds pairs of siblings in a single frame. Takes subdataframe of full tracking
    dataframe as input, and maps to a new coordinate space that includes a measure
    of the 'antiparallel-ness' of velocities to select for sibling nuclei, using
    trackpy to link siblings using this coordinate space. The 'antiparallel-ness'
    coordinate can be based on the full velocity vector or on the directions only,
    and can be rescaled to give arbitrary weight relative to spacial proximity
    during tracking. We can also take a collision-based approach, projecting the
    velocity vectors a half-frame and linking by proximity. This can also pass through
    all of the standard trackpy keyword arguments.
    """
    # Exclude all recently divided particles from search
    vel_columns = ["".join(["v_", pos]) for pos in pos_columns]

    finite_velocity = frame_df[vel_columns].notnull().all(axis=1)
    finite_position = frame_df[pos_columns].notnull().all(axis=1)
    finite_coordinates = finite_velocity & finite_position
    recently_divided_parent = (~frame_df["new_particle"]) & (
        frame_df["age"] <= minimum_age
    )
    remove_particles = recently_divided_parent | (~finite_coordinates)
    frame_df = frame_df.drop(frame_df[remove_particles].index)

    if frame_df.empty:
        siblings_array = np.array([], dtype=int)
        return siblings_array

    # Change frame numbers so all old particles are in frame 1 and all new particles are
    # in frame 2. This is so we can directly use trackpy to link siblings by pretending
    # that the new particles are in the next frame and need to be linked.
    frame_df.loc[~frame_df["new_particle"], "frame"] = 1
    frame_df.loc[frame_df["new_particle"], "frame"] = 2

    if antiparallel_coordinate == "collision":
        frame_df[pos_columns] -= 0.5 * frame_df[vel_columns].values

    else:
        # Rescale position coordinates so that mean distance between nuclei in each coordinate
        # direction is unity
        bounding_box = np.array(
            [(frame_df[pos].max() - frame_df[pos].min()) for pos in pos_columns]
        )
        bounding_volume = np.prod(bounding_box)
        mean_separation = (bounding_volume / frame_df.shape[0]) ** (
            1 / len(pos_columns)
        )
        frame_df[pos_columns] /= mean_separation

        # We can now add a coordinate that represents the rescaled velocities such that the
        # mean velocity in each coordinate direction is unity, similar to what we did for the
        # position coordinates. Now, however, invert the velocities of the new particles such
        # that they appear 'closer' to trackpy the more antiparallel their velocities are

        # We can construct the antiparallelism coordinates so that they represent the full
        # velocity vector (taking into consideration their magnitude) or just the directions
        normalization = (
            frame_df[vel_columns]
            .apply(lambda x: x**2)
            .sum(axis=1)
            .apply(lambda x: np.sqrt(x))
        )

        if antiparallel_coordinate == "direction":
            pass
        elif antiparallel_coordinate == "velocity":
            normalization = normalization.mean()
        else:
            raise Exception("`antiparallel_coordinate` option not recognized.")

        frame_df[vel_columns] = frame_df[vel_columns].divide(normalization, axis=0)

        # We can further rescale the antiparallelism measure to weigh spatial proximity vs
        # velocity antiparallelism as needed
        frame_df[vel_columns] *= antiparallel_weight
        frame_df.loc[frame_df["new_particle"], vel_columns] *= -1

        pos_columns = pos_columns + vel_columns

    linked_siblings = trackpy.link_df(
        frame_df,
        search_range=search_range,
        pos_columns=pos_columns,
        t_column="frame",
        **kwargs,
    )

    # We can finally extract pairs of siblings from the linking
    siblings_mask = linked_siblings.duplicated(subset=["particle"], keep=False)
    num_siblings = siblings_mask.sum()

    siblings_array = np.asarray(
        np.split(
            linked_siblings[siblings_mask]
            .sort_values(["particle", "frame"], ascending=[True, False])
            .index,
            2,
        )
    ).reshape((int(num_siblings / 2), 2))

    return siblings_array


def _assign_siblings(
    tracked_df,
    pos_columns,
    search_range,
    antiparallel_coordinate,
    antiparallel_weight,
    min_track_length,
    image_dimensions,
    exclude_border,
    minimum_age,
    **kwargs,
):
    """
    Returns an (2, n)-shape ndarray with each element along the 0-th axis corresponding
    respectively to the index in `tracked_dataframe` of a new track and to the index
    of its sibling. This can then be used to construct lineages. Fraction
    `exclude_border` of the image in each border is excluded from the search for
    new particles to assign siblings to, and tracks under some age limit `minimum_age`
    are excluded from the search for siblings.
    """
    # Mark new tracks
    tracks_start(
        tracked_df,
        pos_columns,
        min_track_length=min_track_length,
        modify_df=True,
        image_dimensions=image_dimensions,
        exclude_border=exclude_border,
        output=False,
    )

    tracked_siblings_df = tracked_df.copy()

    # Keep track of age of particles for filtering which nuclei we allow to be
    # linked (i.e. particles that recently divided will not be included in the
    # search for new divisions).
    tracked_siblings_df["age"] = 0
    particle_groups = tracked_siblings_df.groupby("particle")
    for _, particle in particle_groups:
        initial_frame = particle["frame"].min()
        particle_index = particle.index
        tracked_siblings_df.loc[particle_index, "age"] = (
            tracked_siblings_df["frame"].loc[particle_index] - initial_frame
        )

    frames = tracked_siblings_df["frame"].sort_values().unique()
    siblings_array = []
    for frame in frames:
        frame_df = tracked_siblings_df[tracked_siblings_df["frame"] == frame].copy()
        frame_siblings = _assign_siblings_frame(
            frame_df,
            pos_columns,
            search_range,
            antiparallel_coordinate,
            antiparallel_weight,
            minimum_age,
            **kwargs,
        )
        if not frame_siblings.size == 0:
            siblings_array.append(frame_siblings)

            # Update the age-tracking column
            for sibling_pair in frame_siblings:
                older_sibling = sibling_pair[1]
                parent_particle = tracked_siblings_df.loc[older_sibling, "particle"]
                new_initial_age = tracked_siblings_df.loc[older_sibling, "age"]
                older_sibling_mask = (
                    tracked_siblings_df["particle"] == parent_particle
                ) & (tracked_siblings_df["age"] >= new_initial_age)
                tracked_siblings_df.loc[older_sibling_mask, "age"] = (
                    tracked_siblings_df.loc[older_sibling_mask, "age"] - new_initial_age
                )

    if len(siblings_array) > 0:
        siblings_array = np.concatenate(siblings_array)

    return siblings_array


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
    antiparallel_coordinate="direction",
    antiparallel_weight=1,
    min_track_length=1,
    image_dimensions=None,
    exclude_border=None,
    minimum_age=5,
    **kwargs,
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
    :type tracked_dataframe: pandas DataFrame
    :param pos_columns: Name of columns in segmentation_df containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param float search_range_mitosis: Search range in transformed coordinate space
        (spatial coordinates in `pos_columns` renormalized so that the mean distance
        between nuclei in each frame is set to unity + one extra coordinate for each
        of the spatial coordinates used to evaluate how antiparallel the velocities
        of candidate siblings are, either using direction vectors or the full
        velocity vectors.
    :param antiparallel_coordinate: Selects which approach to use to construct the
        measure of anti-parallel velocities, with `velocity` using the full velocity
        vector (i.e. it expects siblings to move away from each other at the same
        speed in the scope frame) normalized such that the mean velocity of all
        particles in the frame in each coordinate is unity, `direction` using
        direction vectors, and 'collision' projecting velocities back a half-frame
        and linking by proximity.
    :type antiparallel_coordinate: {'velocity', 'direction', 'collision'}
    :param float antiparallel_weight: Rescales the coordinate space so that an
        arbitrary weight can be placed on spatial proximity vs having
        antiparallel velocities during the search for sibling nuclei, with 1
        corresponding to no rescaling other than that described above (i.e. with
        mean separation between nuclei and velocities in all specified coordinates
        rescaled to unity).
    :param int min_track_length: Minimum number of frames that a new track must
        span to be considered. This helps reduce spurious detections.
    :param image_dimensions: Image dimension in each respective coordinate of
        `pos_columns`.
    :type image_dimensions: Array of integers.
    :param exclude_border: Fraction of border in each position coordinate
        to exclude when searching for new particles to assign siblings to.
    :type exclude_border: float or array of floats for each coordinate.
    :param int minimum_age: Minimum number of frame that a particle needs to have
        existed for to be considered during the search for siblings. This helps
        cut down on single nuclei being assigned several divisions due to proximity
        with other dividing nuclei.
    :return: Copy of `tracked_dataframe` with particle identities corresponding to
        lineage and an added column with parent identities.
    :rtype: pandas DataFrame

    .. note::
        This function can also pass through any of the `trackpy.link_df`
        arguments.
    """
    mitosis_dataframe = tracked_dataframe.copy()

    # Construct array of siblings
    sibling_array = _assign_siblings(
        tracked_dataframe,
        pos_columns,
        search_range_mitosis,
        antiparallel_coordinate,
        antiparallel_weight,
        min_track_length,
        image_dimensions,
        exclude_border,
        minimum_age,
        **kwargs,
    )

    # Assign parent by indexing over dataframe
    parent_index_array = _assign_parents(mitosis_dataframe, sibling_array)

    # Assign new labels to siblings and clip parent track
    _relabel_sibling(mitosis_dataframe, sibling_array)

    # Add parents to first frame of sibling pairs
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


def tracks_to_napari(viewer, dataframe, name="tracks", output=False):
    """
    Uses tracking and lineage information in `mitosis_dataframe` to construct
    tracks datastructures compatible with napari tracks layer. If output is requested,
    will also return these data structures.

    :param viewer: napari viewer.
    :type viewer: napari viewer object
    :param dataframe: DataFrame of measured features after tracking with
        :func:`~link_dataframe` and lineage construction with `construct_lineage`.
    :type dataframe: pandas DataFrame
    :param str name: Name of the `tracks` layer in napari.
    :param bool output: If True, returns the napari-compatible datastructures used
        to visualize tracks.
    :return: If `output = True`, returns lineage dictionary and parent properties
        used to visualize tracks in napari.
    :rtype: {tuple, None}
    """
    mitosis_dataframe = dataframe.copy()
    mitosis_dataframe["frame"] = mitosis_dataframe["frame"] - 1

    tracks = mitosis_dataframe.loc[:, ["particle", "frame", "z", "y", "x"]].to_numpy()

    lineage = {}
    for _, family in mitosis_dataframe.groupby("parent"):
        parent = family["parent"].values[0]
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
