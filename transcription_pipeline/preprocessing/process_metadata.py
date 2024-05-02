import numpy as np


def extract_time(frame_metadata):
    """
    Returns a function that maps a frame number and z-coordinate to the time in
    seconds after the start of the imaging sessions at which that coordinate was
    imaged.

    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :return: tuple of function object taking frame number and z-position as arguments
        and returning an imaging time in seconds, and time between frames i.e. of
        form tuple(function of form time_func(frame_number, z_position),
        time_between_frames).
    :rtype: tuple

    .. note::

        This function will try to estimate the time at which a coordinate was imaged.
        If the scope records the imaging time for each z-slice in the metadata, it will
        simply use the corresponding value. Some scopes, however, only record the
        time at the start of each z-stack - in this case it will interpolate the
        time.

    """
    # Check if imaging time is encoded on z-slice or frame basis
    frame_times = frame_metadata["t_s"]
    first_frame_times = frame_times[0]
    time_by_frame = np.sum(first_frame_times - first_frame_times[0]) == 0

    time_between_frames = np.diff(frame_times[:, 0])
    # Use median difference between frames to estimate time to scan over a z-stack
    # (this avoids issues with datasets made of multiple series).
    frame_scan_time = np.median(time_between_frames)

    if time_by_frame:
        num_slices = frame_metadata["z"].max() + 1
        time_per_slice = frame_scan_time / num_slices

        time_func = (
            lambda frame_number, z_position: frame_times[frame_number - 1, 0]
            + time_per_slice * z_position
        )

    else:

        def time_func(frame_number, z_position):
            """
            Helper function to calculate imaging time from the frame number and
            position in the z-stack.
            """
            return frame_times[frame_number - 1, int(z_position)]

    return time_func, frame_scan_time


def extract_renormalized_frame(frame_metadata):
    """
    Returns a function that maps a frame number and z-coordinate to the time in
    units of number of z-stack scans after the start of the imaging sessions - this
    is used rather than absolute time in seconds by trackpy.

    :param dict frame_metadata: Dictionary of frame-by-frame metadata for all files and
        series in a dataset.
    :return: Function object taking frame number and z-position as arguments and
        returning an imaging time number of z-stack scan times.
    :rtype: function of form time_func(frame_number, z_position)

    .. note::

        This function will try to estimate the time at which a coordinate was imaged.
        If the scope records the imaging time for each z-slice in the metadata, it will
        simply use the corresponding value. Some scopes, however, only record the
        time at the start of each z-stack - in this case it will interpolate the
        time.

    """
    time_func, time_between_frames = extract_time(frame_metadata)

    def frame_time_func(frame_number, z_position):
        """
        Helper function to calculate imaging time, in units of the frame scanning
        time, from the frame number and position in the z-stack.
        """
        return int(time_func(frame_number, z_position) / time_between_frames)

    return frame_time_func
