import os
import sys
import glob
from datetime import datetime
from itertools import groupby
from pathlib import Path
import numpy as np
import pims
from scipy.io import savemat
from skimage.io import imsave
import numbers
import zarr
import deprecation
from scipy.signal import peak_prominences
import warnings


def _set_java_environment_path():
    """
    Sets the `JAVA_HOME` environment variable that points the javabridge
    used by the PIMS Bioformats reader to the environment folder.
    """
    environment_path = Path(sys.prefix)
    os.environ["JAVA_HOME"] = str(environment_path)

    print("".join(["`JAVA_HOME` environment variable set to ", str(environment_path)]))


_set_java_environment_path()


def extract_global_metadata(metadata_object):
    """
    Helper function that extracts a dictionary of the global metadata
    (i.e. not the frame-by-frame metadata) from a MetadataRetrieve object
    as given by the PIMS bioformats reader for a single file.

    :param metadata_object: MetadataRetrieve object output by PIMS bioformats
        reader for single file.
    :type metadata_object: MetadataRetrieve object.
    :return: Dictionary of global metadata values. If the metadata field has
        potentially distinct values for different series and channels within
        the file, the corresponding key is a list indexed by series and
        channel respectively.
    :rtype: dict
    """
    metadata_retrieve = str(metadata_object).split(", ")
    metadata_retrieve[0] = metadata_retrieve[0].split(": ")[1]

    num_series = metadata_object.ImageCount()
    global_metadata = {}
    for field in metadata_retrieve:
        # First write metadata as dict of function objects
        field_data_func = getattr(metadata_object, field)

        # Convert each function object into a list of values for each series
        # and channel, indexed in that order.

        # Using exceptions for flow control is not ideal, but the PIMS wrappers
        # use variable-length arguments so I don't have a good way of
        # inspecting the argument structure for each metadata object a priori.
        # The key here is that TypeError is only raised when the incorrect
        # number of arguments/indices are given, and a java exception
        # (caught by a general Exception except block) is raised when the
        # correct number are arguments are given but the indices exceed
        # the size of the metadata object.
        try:
            field_data = field_data_func()
        except TypeError:
            try:
                field_data = []
                for i in range(num_series):
                    field_data.append(field_data_func(i))
            except TypeError:
                field_data = []
                for i in range(num_series):
                    num_channels = metadata_object.ChannelCount(i)
                    series_field_data = []
                    for j in range(num_channels):
                        # This handles the weird argument shape that doesn't
                        # match channel structure for ChannelAnnotationRefCount
                        try:
                            series_field_data.append(field_data_func(i, j))
                        except Exception:
                            try:
                                series_field_data.append(field_data_func(i, j, 0))
                            except Exception:
                                pass

                    if len(series_field_data) == 1:
                        field_data.append(series_field_data[0])
                    else:
                        field_data.append(series_field_data)

        global_metadata[field] = field_data

    return global_metadata


def all_equal(iterable):
    """
    Checks if all the elements of an iterable are identical.

    :param iterable: Any object that can be iterated over.
    :return: True if all elements of input are identical, False otherwise.
    :rtype: bool
    """

    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def check_metadata(metadata_dict, ignore_fields=[]):
    """
    Checks dictionary of global metadata as output by
    :func:`~extract_global_metadata` for inconsistencies and flattens
    fields with no inconsistencies. Throws a warning for each inconsistent
    metadata field, except for those listed in ignore_fields.

    :param dict metadata_dict: Dictionary of global metadata.
    :param list ignore_fields: List of metadata fields to leave uncollapsed
        even if they are consistent across series.
    :return: Cleaned up dictionary of global metadata, with consistent fields
        flattened with removed duplicates.
    :rtype: dict
    """

    cleaned_dict = {}
    for field in metadata_dict:
        metadata_value = metadata_dict[field]
        try:
            try:
                # Check consistency of settings between series
                if all_equal(metadata_value) and field not in ignore_fields:
                    cleaned_dict[field] = metadata_value[0]
                else:
                    cleaned_dict[field] = metadata_value
                    raise Warning(
                        "".join(
                            [
                                "The series in {0} have inconsistent ",
                                "{1}, check your imaging settings and ",
                                "metadata.",
                            ]
                        ).format(metadata_dict["ImageName"], field)
                    )
            except Warning as message:
                if field not in ignore_fields:
                    print(message)

        except TypeError:
            cleaned_dict[field] = metadata_value

    return cleaned_dict


def collate_global_metadata(
    input_global_metadata,
    num_channels,
    num_series_total,
    trim_series,
    fields_num_channels,
    fields_num_series,
    fields_num_series_channels,
    fields_explicit,
):
    """
    Helper function that uses the metadata from the files being processed
    to write a consistend global metadata dictionary for the collated file.

    :param dict input_global_metadata: Clean up metadata dictionary for an
        extracted dataset, as output by :func:`~check_metadata`.
    :param int num_channels: Number of imaging channels.
    :param int num_series_total: Total number of series across all data files.
    :param bool trim_series: If True, deletes the last frame of each series.
        This should be used when acquisition was stopped in the middle of a
        z-stack.
    :param list fields_num_channels: List of metadata fields indexed by
        channel.
    :param list fields_num_series: List of metadata fields indexed by series.
    :param list fields_num_series_channels: List of metadata fields indexed
        by series and channels respectively.
    :param list fields_explicit: List of metadata fields that will be handled
        explicitly - this just avoids multiple writings to the same dict key.
    :return: Dictionary of global metadata for the exported (collated) file.
    :rtype: dict
    """
    output_global_metadata = []
    for i in range(num_channels):
        channel_global_metadata = {}
        for field in input_global_metadata:
            if field in fields_num_channels:
                if num_channels > 1:
                    channel_global_metadata[field] = (input_global_metadata[field])[i]
                else:
                    channel_global_metadata[field] = input_global_metadata[field]
            elif field in fields_num_series:
                channel_global_metadata[field] = (input_global_metadata[field])[0][0]
            elif field in fields_num_series_channels:
                if num_channels > 1:
                    channel_global_metadata[field] = (input_global_metadata[field])[0][
                        0
                    ][i]
                else:
                    channel_global_metadata[field] = (input_global_metadata[field])[0][
                        0
                    ]
            elif field not in fields_explicit:
                if isinstance(input_global_metadata[field], list):
                    channel_global_metadata[field] = "inconsistent_metadata"
                else:
                    channel_global_metadata[field] = input_global_metadata[field]

        (channel_global_metadata)["ImageName"] = "collated_dataset_ch{:02d}".format(i)
        (channel_global_metadata)["ImageCount"] = 1
        (channel_global_metadata)["InstrumentCount"] = 1

        pixel_size_t = input_global_metadata["PixelsSizeT"][0]
        total_timepoints = sum(
            sum(x) if isinstance(x, list) else x for x in pixel_size_t
        )

        pixel_size_z = input_global_metadata["PixelsSizeZ"]

        if isinstance(pixel_size_z, list):
            raise Exception("Inconsistent z-stack size across series.")
        else:
            total_planes = pixel_size_z * total_timepoints

        if trim_series:
            total_timepoints -= num_series_total
            total_planes -= num_series_total * pixel_size_z

        (channel_global_metadata)["PixelsSizeT"] = total_timepoints
        (channel_global_metadata)["PlaneCount"] = total_planes

        output_global_metadata.append(channel_global_metadata)

    return output_global_metadata


def collate_frame_metadata(
    input_frame_metadata,
    input_global_metadata,
    output_global_metadata,
    trim_series,
    num_channels,
    time_delta_from_0,
):
    """
    Helper function that uses the metadata from the files being processed
    to write a consistend frame-by-frame metadata dictionary for the
    collated file.

    :param dict input_frame_metadata; Frame-by-frame metadata dictionary for
        an extracted dataset.
    :param dict input_global_metadata: Cleaned up metadata dictionary for an
        extracted dataset, as output by :func:`~check_metadata`.
    :param dict output_global_metadata: Global metadata dictionary for the
        exported (collated) data file, as output by
        :func:`~collate_global_metadata`.
    :param bool trim_series: If True, deletes the last frame of each series.
        This should be used when acquisition was stopped in the middle of a
        z-stack.
    :param int num_channels: Number of imaging channels.
    :param list time_delta_from_0: List of time difference between the
        start of each series and the start of the first series in the dataset.
        This is used to make an accurate timestamp of the frames in all
        series in a dataset since each series records its timestamps relative
        to its own start time.
    :return: Dictionary of global metadata for the exported (collated) file.
    :rtype: dict
    """
    output_frame_metadata = []
    for i in range(num_channels):
        channel_frame_metadata = {}

        total_planes = (output_global_metadata[i])["PlaneCount"]
        total_timepoints = (output_global_metadata[i])["PixelsSizeT"]
        pixel_size_z = (output_global_metadata[i])["PixelsSizeZ"]
        frame_indices = np.arange(total_planes)
        new_shape = (total_timepoints, pixel_size_z)
        frame_indices = np.reshape(frame_indices, new_shape)
        channel_frame_metadata["frame"] = frame_indices

        channel_frame_metadata["series"] = 0

        def clear_duplicates(x):
            return x[0] if all_equal(x) else "inconsistent_metadata"

        single_value_metadata = [
            "mpp",
            "mppZ",
            "x",
            "y",
            "x_um",
            "y_um",
            "axes",
            "coords",
        ]
        for key in single_value_metadata:
            channel_frame_metadata[key] = clear_duplicates(input_frame_metadata[key])

        c = np.ones(new_shape) * i
        channel_frame_metadata["c"] = c

        channel_frame_metadata["colors"] = (
            clear_duplicates(input_frame_metadata["colors"])
        )[i]

        if trim_series:
            end = -1
        else:
            end = None

        try:
            if num_channels > 1:
                z_list = [
                    (z_original[:end])[:, i, :]
                    for z_original in input_frame_metadata["z"]
                ]
            else:
                z_list = [z_original[:end] for z_original in input_frame_metadata["z"]]
            z = np.concatenate(z_list)
            channel_frame_metadata["z"] = z
        except ValueError:
            z = "inconsistent_metadata"

        try:
            if num_channels > 1:
                t_list = [
                    (t_original[:end])[:, i, :]
                    for t_original in input_frame_metadata["t"]
                ]
            else:
                t_list = [
                    (t_original[:end]) for t_original in input_frame_metadata["t"]
                ]
            t_list_offset = [np.copy(t_series) for t_series in t_list]
            for j in range(1, len(t_list)):
                t_list_offset[j] += t_list_offset[j - 1][-1, -1] + 1
            t = np.concatenate(t_list_offset)
            channel_frame_metadata["t"] = t
        except ValueError:
            t = "inconsistent_metadata"

        try:
            if num_channels > 1:
                t_s_list = [
                    (t_s_original[:end])[:, i, :]
                    for t_s_original in input_frame_metadata["t_s"]
                ]
            else:
                t_s_list = [
                    (t_s_original[:end]) for t_s_original in input_frame_metadata["t_s"]
                ]
            t_s_list_offset = [np.copy(t_s_series) for t_s_series in t_s_list]
            t_s_is_number = issubclass(t_s_list_offset[0].dtype.type, numbers.Number)
            # Checking if t_s is numerical (as opposed to string) since presence of
            # 'None' during partial scans forces conversion to <U32.
            if not t_s_is_number:
                t_s_list_offset[0][t_s_list_offset[0] == "None"] = np.nan
            t_s_list_offset[0] = t_s_list_offset[0].astype(float)

            for j in range(1, len(t_s_list)):
                t_s_is_number = issubclass(
                    t_s_list_offset[j].dtype.type, numbers.Number
                )
                if not t_s_is_number:
                    t_s_list_offset[j][t_s_list_offset[j] == "None"] = np.nan
                t_s_list_offset[j] = (
                    t_s_list_offset[j].astype(float) + time_delta_from_0[j]
                )
            t_s = np.concatenate(t_s_list_offset)
            channel_frame_metadata["t_s"] = t_s
        except ValueError:
            t_s = "inconsistent_metadata"

        output_frame_metadata.append(channel_frame_metadata)

    return output_frame_metadata


def collate_metadata(input_global_metadata, input_frame_metadata, trim_series):
    """
    Helper function that takes the original metadata extracted from each
    series and data file by PIMS (where each dictionary key value is a list
    of the corresponding metadata values for each series, indexed by data file
    and series respectively) and returns cleaned up global and frame-by-frame
    metadata dictionaries consistent with the PIMS format, corresponding
    to the cleaned-up (trimmed and collated) dataset.

    :param dict input_global_metadata: Dictionary of global metadata for all
        files and series in a dataset. Make sure that the global metadata
        has been run through :func:`~check_metadata` since this function
        raises an exception if the ChannelCount and PixelSizeZ metadata are
        still given as lists.
    :param dict input_frame_metadata: Dictionary of frame-by-frame metadata
        for all files and series in a dataset.
    :param bool trim_series: If True, deletes the last frame of each series.
        This should be used when acquisition was stopped in the middle of a
        z-stack.
    :return: Tuple(output_global_metadata, output_frame_metadata) of
        lists of dictionaries, each list element corresponding to a channel,
        matching the global and frame-by-frame metadata respectively of the
        collated dataset.
    :rtype: Tuple of lists of dictionaries.
    """
    # First, handle the global metadata
    # If the metadata in the original files is consistent, the following
    # fields should be lists of the same length as the number of channels
    num_channels = input_global_metadata["ChannelCount"]
    if isinstance(num_channels, list):
        raise Exception("Number of channels inconsistent across dataset.")

    image_count = input_global_metadata["ImageCount"]
    if isinstance(image_count, list):
        num_series_total = sum(image_count)
    else:
        num_series_total = image_count

    fields_num_channels = [
        "ChannelAnnotationRefCount",
        "ChannelColor",
        "ChannelName",
        "ChannelPinholeSize",
        "ChannelSamplesPerPixel",
        "DetectorAmplificationGain",
        "DetectorAnnotationRefCount",
        "DetectorGain",
        "DetectorID",
        "DetectorSettingsID",
        "DetectorType",
        "DetectorZoom",
        "DichroicAnnotationRefCount",
        "DichroicID",
        "DichroicModel",
        "LaserID",
        "LaserLaserMedium",
        "LaserModel",
        "LaserType",
        "LaserWavelength",
        "LightPathAnnotationRefCount",
        "LightPathEmissionFilterRefCount",
        "LightPathExcitationFilterRefCount",
        "LightSourceAnnotationRefCount",
        "LightSourceType",
        "PlaneAnnotationRefCount",
        "PlaneDeltaT",
        "PlanePositionX",
        "PlanePositionY",
        "PlanePositionZ",
        "PlaneTheC",
        "PlaneTheT",
        "PlaneTheZ",
    ]

    # The following fields should be lists of the same length as the
    # number of series.
    fields_num_series = [
        "ImageID",
        "ImageInstrumentRef",
        "InstrumentID",
        "ObjectiveID",
        "ObjectiveSettingsID",
        "PixelsID",
    ]

    # The following fields are nested lists, indexed first by series then by
    # channel.
    fields_num_series_channels = [
        "ChannelID",
        "ImageAcquisitionDate",
    ]

    # The following fields are taken care of explicitly above to match the
    # dataset collation.
    fields_explicit = [
        "ImageName",
        "ImageCount",
        "InstrumentCount",
        "PixelsSizeT",
        "PlaneCount",
    ]

    output_global_metadata = collate_global_metadata(
        input_global_metadata,
        num_channels,
        trim_series,
        num_series_total,
        fields_num_channels,
        fields_num_series,
        fields_num_series_channels,
        fields_explicit,
    )

    # Now we handle the frame-by-frame metadata
    acquisition_times = input_global_metadata["ImageAcquisitionDate"]
    # Flatten list so it is indexed by series
    acquisition_times = [item for sublist in acquisition_times for item in sublist]
    acquisition_times = [
        datetime.strptime(series_time, "%Y-%m-%dT%H:%M:%S")
        for series_time in acquisition_times
    ]
    time_delta_from_0 = [
        (acquisition_times[i] - acquisition_times[0]).total_seconds()
        for i in range(len(acquisition_times))
    ]

    output_frame_metadata = collate_frame_metadata(
        input_frame_metadata,
        input_global_metadata,
        output_global_metadata,
        trim_series,
        num_channels,
        time_delta_from_0,
    )

    return (output_global_metadata, output_frame_metadata)


def z_cross_correlation_shift(stack1, stack2, peak_window=3, min_prominence=0.2):
    """
    Used the normalized cross-correlation in the z-axis to find the shift in z between
    two z-stacks.

    :param stack1: Z-stack overlapping with `stack2` such that the normalized correlation
        peak can be used to estimate the shift in z between `stack1` and `stack2`.
    :type stack1: Numpy array.
    :param stack2: Z-stack overlapping with `stack1` such that the normalized correlation
        peak can be used to estimate the shift in z between `stack1` and `stack2`.
    :type stack2: Numpy array
    :param int peak_window: Window around proposed peak index in normalized correlation
        computed for different z-shifts to consider when computing the centroid of the
        peak of the normalized correlation.
    :param float min_prominence: Minimum prominence of a peak for it to be considered
        a true maxima of the normalized correlation. See documentation for
        `scipy.signal.peak_prominences`.
    :return: Estimated shift in pixels (sub-pixel approximated using centroid of
        normalized correlation peak) between `stack1` and `stack2`.
    :rtype: float
    """
    # Check stack dimensions
    if stack1.shape != stack2.shape:
        raise ValueError("Stacks need to be the same shape.")

    # Normalize images
    stack1 = (stack1 - stack1.mean()) / stack1.std()
    stack2 = (stack2 - stack2.mean()) / stack2.std()

    # Calculate correlation function for all possible overlaps along z
    num_z_slices = stack1.shape[0]
    offsets = np.zeros(2 * num_z_slices - 1)
    for i in range(num_z_slices):
        offsets[i] = (stack1[-(i + 1) :] * stack2[: (i + 1)]).mean()
        offsets[-(i + 1)] = (stack2[-(i + 1) :] * stack1[: (i + 1)]).mean()

    proposed_peak = offsets.argmax()

    # Check prominence of peak
    prominence = peak_prominences(offsets, [proposed_peak])[0]

    if prominence > min_prominence:
        # Restrict attention to points near proposed peak
        considered_indices = np.arange(
            proposed_peak - peak_window, proposed_peak + peak_window + 1
        )
        considered_offsets = offsets[considered_indices]

        # Find centroid of considered peak
        centroid = (
            considered_indices * considered_offsets
        ).sum() / considered_offsets.sum()

        # Compute shift in images
        shift_pixels = centroid - num_z_slices + 1

    else:
        shift_pixels = np.nan
    return shift_pixels


def import_dataset(
    name_folder, trim_series, peak_window=3, min_prominence=0.2, registration_channel=-1
):
    """
    Imports and collates the data files in the name_directory, and generates
    metadata dictionaries for the origininal files metadata and new metadata
    corresponding to the collated file.

    :param str name_folder: Path to name folder containing data files.
    :param bool trim_series: If True, deletes the last frame of each series.
        This should be used when acquisition was stopped in the middle of a
        z-stack.
    :param int peak_window: Window around proposed peak index in normalized correlation
        computed for different z-shifts to consider when computing the centroid of the
        peak of the normalized correlation.
    :param float min_prominence: Minimum prominence of a peak for it to be considered
        a true maxima of the normalized correlation. See documentation for
        `scipy.signal.peak_prominences`.
    :param int registration_channel: Index of channel to use for registration of z-shift
        between stacks - usually works better with channels with more structure. By default
        uses last channel.
    :return: Tuple(channels_full_dataset, original_global_metadata,
           original_frame_metadata, export_global_metadata,
           export_frame_metadata)
           * ``channels_full_dataset``: list of numpy arrays, with each
           element of the list being a collated dataset for a given channel.
           * ``original_global_metadata``: dictionary of global metadata
           for all files and series in a dataset.
           * ``original_frame_metadata``: dictionary of frame-by-frame metadata
           for all files and series in a dataset.
           * ``export_global_metadata``: list of dictionaries of global
           metadata for the collated dataset, with each element of the list
           corresponding to a channel.
           * ``export_frame_metadata``: list of dictionaries of frame-by-frame
           metadata for the collated dataset, with each element of the list
           corresponding to a channel.
           * ``series_splits``: list of first frame of each series. This is useful
           when stitching together z-coordinates to improve tracking when the z-stack
           has been shifted mid-imaging.
           *``series_shifts``: list of estimated shifts in pixels (sub-pixel
           approximated using centroid of normalized correlation peak) between stacks
           at interface between separate series - this quantifies the shift in the
           z-stack.
    :rtype: Tuple
    """
    name_folder_path = Path(name_folder)
    dataset_name = name_folder_path.name
    file_path = name_folder_path / "".join([dataset_name, "*"])
    file_list = glob.glob(str(file_path))
    file_list.sort()

    # Metadata fields to ignore during consistency checks (these fields
    # typically vary from series to series).
    ignore_fields = [
        "ChannelID",
        "FilterID",
        "ImageAcquisitionDate",
        "ImageID",
        "ImageInstrumentRef",
        "ImageName",
        "InstrumentID",
        "LightPathEmissionFilterRef",
        "ObjectiveID",
        "ObjectiveSettingsID",
        "PixelsID",
        "PixelsSizeT",
        "PlaneCount",
    ]

    # We pull the individual data files into a list as pipeline objects.
    # Pulling the frame-by-frame metadata is deferred until the collation step
    # since it forces a (slower) read from disk.
    data = []  # One element per series
    num_frames_series = []  # One element per series
    metadata_list = []  # One element per file

    for file in file_list:
        # Open a reader for the first series of the file
        series = pims.Bioformats(file, read_mode="jpype", series=0)
        try:
            series.bundle_axes = "tczyx"
            multichannel = True
        except ValueError:
            series.bundle_axes = "tzyx"
            multichannel = False

        data.append(series)
        num_frames_series.append((series.shape)[1])

        # Extract metadata for each file and do consistency checks
        file_metadata = extract_global_metadata(series.metadata)
        checked_file_metadata = check_metadata(file_metadata, ignore_fields)
        metadata_list.append(checked_file_metadata)

        # Open subsequent readers for each series within the file
        num_series = checked_file_metadata["ImageCount"]
        for i in range(1, num_series):
            series = pims.Bioformats(file, read_mode="jpype", series=i)
            if multichannel:
                series.bundle_axes = "tczyx"
            else:
                series.bundle_axes = "tzyx"
            data.append(series)
            num_frames_series.append((series.shape)[1])

    # We check for imaging settings consistency between files (the previous
    # block should already have checked for consistency between series).
    full_dataset_global_metadata = {}
    for file_metadata in metadata_list:
        for field in file_metadata:
            if field in full_dataset_global_metadata:
                full_dataset_global_metadata[field].append(file_metadata[field])
            else:
                full_dataset_global_metadata[field] = [file_metadata[field]]

    original_global_metadata = check_metadata(
        full_dataset_global_metadata, ignore_fields
    )

    num_frames_series = np.array(num_frames_series)  # Numpy array more
    # convenient for later
    # slicing

    num_frames = np.sum(num_frames_series)
    num_series = num_frames_series.size
    if trim_series:
        num_frames = num_frames - num_series
        num_frames_series = num_frames_series - 1
        end = -1
    else:
        end = None

    series_shape = data[0].shape
    dataset_shape = (
        num_frames,
        *series_shape[2:],
    )

    dtype = original_global_metadata["PixelsType"]
    full_dataset = np.empty(dataset_shape, dtype=dtype)

    original_frame_metadata = {}  # One element per series

    # Collate the dataset and pull frame-by-frame metadata
    series_start = 0
    series_splits = []
    for i, _ in enumerate(data):
        series_end = series_start + num_frames_series[i]
        full_dataset[series_start:series_end] = data[i][0][:end]
        series_frame_metadata = data[i][0].metadata

        for frame_key in series_frame_metadata:
            if frame_key in original_frame_metadata:
                original_frame_metadata[frame_key].append(
                    series_frame_metadata[frame_key]
                )
            else:
                original_frame_metadata[frame_key] = [series_frame_metadata[frame_key]]

        series_start = series_end
        series_splits.append(series_start)

    series_splits = series_splits[:-1] # Remove end of last series

    # Estimate shift in z-stacks
    series_shifts = []
    for series_start in series_splits:
        if multichannel:
            post_shift = full_dataset[series_start][registration_channel]
            pre_shift = full_dataset[series_start - 1][registration_channel]
        else:
            post_shift = full_dataset[series_start]
            pre_shift = full_dataset[series_start - 1]
        series_shifts.append(
            z_cross_correlation_shift(
                pre_shift,
                post_shift,
                peak_window=peak_window,
                min_prominence=min_prominence,
            )
        )

    # Create metadata dicts for the collated data
    export_global_metadata, export_frame_metadata = collate_metadata(
        original_global_metadata, original_frame_metadata, trim_series
    )

    # Close the readers
    for reader in data:
        reader.close()

    # Split datasets into channels
    num_channels = original_global_metadata["ChannelCount"]
    channels_full_dataset = []
    for i in range(num_channels):
        if num_channels > 1:
            channels_full_dataset.append(full_dataset[:, i, ...])
        else:
            channels_full_dataset.append(full_dataset)

    return (
        channels_full_dataset,
        original_global_metadata,
        original_frame_metadata,
        export_global_metadata,
        export_frame_metadata,
        series_splits,
        series_shifts,
    )


@deprecation.deprecated(
    details="Deprecated in favor of `pipeline.DataImport` class wrapper."
)
def import_save_dataset(name_folder, *, trim_series, mode="tiff", chunks=False):
    """
    Imports and collates the data files in the name_directory, and generates
    metadata dictionaries for the original files metadata and new metadata
    corresponding to the collated file. These metadata dicts are saved to
    a 'collated_dataset' folder in the name_folder as .mat files and the
    collated dataset is saved into separate tiff files for each channel in the
    same folder.

    :param str name_folder: Path to name folder containing data files.
    :param bool trim_series: If True, deletes the last frame of each series.
        This should be used when acquisition was stopped in the middle of a
        z-stack.
    :param mode: Format to save and return data array.
    :type mode: {'tiff', 'zarr'}
    :param chunks: Chunk shape for zarr storage. If True, will be guessed from shape
        and dtype. If False, will be set to shape, i.e., single chunk for the whole
        array. If an int, the chunk size in each dimension will be given by the
        value of chunks. Default is False.
    :type chunks: bool, int or tuple of ints, optional.
    :return: Tuple(channels_full_dataset, original_global_metadata,
        original_frame_metadata, export_global_metadata,export_frame_metadata)
        * ``channels_full_dataset``: list of numpy (if mode='tiff') or zarr arrays
        (if mode='zarr'), with each element of the list being a collated dataset
        for a given channel.
        * ``original_global_metadata``: dictionary of global metadata
        for all files and series in a dataset.
        * ``original_frame_metadata``: dictionary of frame-by-frame metadata
        for all files and series in a dataset.
        * ``export_global_metadata``: list of dictionaries of global
        metadata for the collated dataset, with each element of the list
        corresponding to a channel.
        * ``export_frame_metadata``: list of dictionaries of frame-by-frame
        metadata for the collated dataset, with each element of the list
        corresponding to a channel.
    :rtype: Tuple of dicts
    """
    (
        channels_full_dataset,
        original_global_metadata,
        original_frame_metadata,
        export_global_metadata,
        export_frame_metadata,
    ) = import_dataset(name_folder, trim_series)

    # Make collated_dataset directory if it doesn't exist
    name_path = Path(name_folder)
    collated_path = name_path / "collated_dataset"
    collated_path.mkdir(exist_ok=True)

    global_metadata_path = collated_path / "original_global_metadata.mat"
    savemat(global_metadata_path, original_global_metadata)

    # Convert frame metadata to numpy object-type arrays to help with
    # saving to file later.
    original_frame_metadata_object = {}
    for frame_key in original_frame_metadata:
        original_frame_metadata_object[frame_key] = np.asanyarray(
            original_frame_metadata[frame_key], dtype=object
        )

    frame_metadata_path = collated_path / "original_frame_metadata.mat"
    savemat(frame_metadata_path, original_frame_metadata_object)

    for i, channel_data in enumerate(channels_full_dataset):
        # Save metadata to file
        collated_global_path = (
            collated_path / "collated_global_metadata_ch{:02d}.mat".format(i)
        )
        savemat(collated_global_path, export_global_metadata[i])

        collated_frame_path = (
            collated_path / "collated_frame_metadata_ch{:02d}.mat".format(i)
        )
        savemat(
            collated_frame_path,
            export_frame_metadata[i],
        )
        if mode == "tiff":
            # Save data to file for each channel
            filename = "".join([(export_global_metadata[i])["ImageName"], ".tiff"])
            collated_data_path = collated_path / filename
            imsave(collated_data_path, channel_data, plugin="tifffile")

        elif mode == "zarr":
            # Save data to file for each channel
            filename = "".join([(export_global_metadata[i])["ImageName"], ".zarr"])
            collated_data_path = collated_path / filename

            # Convert to zarr
            store = zarr.storage.DirectoryStore(collated_data_path)
            channel_data = zarr.creation.array(channel_data, chunks=chunks, store=store)
            store.close()

            channels_full_dataset[i] = channel_data

        else:
            raise Exception("Save mode not recognized.")

    return (
        channels_full_dataset,
        original_global_metadata,
        original_frame_metadata,
        export_global_metadata,
        export_frame_metadata,
    )


def import_full_embryo(name_folder, name):
    """
    Imports the FullEmbryo data files in the name_directory used to assign position of
    the imaging window along the embryo AP axis, and generates metadata dictionaries for
    the origininal files metadata and new metadata corresponding to the respective
    channels.

    :param str name_folder: Path to name folder containing data files.
    :param str name: Name of file pattern to look for inside the FullEmbryo folder
        (e.g. `Mid*`).
    :return: Tuple(channels_full_dataset, original_global_metadata,
           original_frame_metadata, export_global_metadata,
           export_frame_metadata)
           * ``channels_dataset``: list of numpy arrays, with each
           element of the list being a collated FullEmbryo dataset for a given channel.
           * ``original_global_metadata``: dictionary of global metadata
           for a FullEmbryo dataset.
           * ``original_frame_metadata``: dictionary of frame-by-frame metadata
           for all files and series in a dataset.
           * ``export_global_metadata``: list of dictionaries of global
           metadata for the collated dataset, with each element of the list
           corresponding to a channel.
           * ``export_frame_metadata``: list of dictionaries of frame-by-frame
           metadata for the collated dataset, with each element of the list
           corresponding to a channel.
    :rtype: Tuple
    """
    name_folder_path = Path(name_folder)
    file_path = name_folder_path / "FullEmbryo" / name
    file_list = glob.glob(str(file_path))

    if len(file_list) > 1:
        raise Exception("Multiple Mid images found in folder.")
    else:
        file = file_list[0]

    # Pull the data file as a pipeline object
    file_image = pims.Bioformats(file, read_mode="jpype")
    try:
        file_image.bundle_axes = "czyx"
        multichannel = True
    except ValueError:
        file_image.bundle_axes = "zyx"
        multichannel = False

    file_metadata = file_image.metadata
    original_global_metadata = extract_global_metadata(file_metadata)
    for field in original_global_metadata:
        if isinstance(original_global_metadata[field], list):
            if len(original_global_metadata[field]) == 1:
                original_global_metadata[field] = original_global_metadata[field][0]

    image = np.asarray(file_image[0])
    original_frame_metadata = file_image[0].metadata

    num_channels = original_global_metadata["ChannelCount"]

    if num_channels > 1:
        # Split the data into a list indexed by channel
        channels_dataset = [image[i] for i in range(num_channels)]

        # Split the frame metadata into a list indexed by channel
        channel_fields = [
            "frame",
            "colors",
            "c",
            "z",
        ]

        export_frame_metadata = []
        for i in range(num_channels):
            channel_frame_metadata = {}
            for field in original_frame_metadata:
                if field in channel_fields:
                    channel_frame_metadata[field] = original_frame_metadata[field][i]
                else:
                    channel_frame_metadata[field] = original_frame_metadata[field]
            channel_frame_metadata["axes"] = channel_frame_metadata["axes"][1:]
            export_frame_metadata.append(channel_frame_metadata)

        # Split the global metadata into a list indexed by channel
        global_channel_fields = [
            "ChannelID" "ChannelAnnotationRefCount",
            "ChannelColor",
            "ChannelName",
            "ChannelPinholeSize",
            "ChannelSamplesPerPixel",
            "DetectorAmplificationGain",
            "DetectorAnnotationRefCount",
            "DetectorGain",
            "DetectorID",
            "DetectorSettingsID",
            "DetectorType",
            "DetectorZoom",
            "DichroicAnnotationRefCount",
            "DichroicID",
            "DichroicModel",
            "LaserID",
            "LaserLaserMedium",
            "LaserModel",
            "LaserType",
            "LaserWavelength",
            "LightPathAnnotationRefCount",
            "LightPathEmissionFilterRefCount",
            "LightPathExcitationFilterRefCount",
            "LightSourceAnnotationRefCount",
            "LightSourceType",
            "PlaneAnnotationRefCount",
            "PlaneDeltaT",
            "PlanePositionX",
            "PlanePositionY",
            "PlanePositionZ",
            "PlaneTheC",
            "PlaneTheT",
            "PlaneTheZ",
        ]

        export_global_metadata = []
        for i in range(num_channels):
            channel_global_metadata = {}
            for field in original_global_metadata:
                if field in global_channel_fields:
                    channel_global_metadata[field] = original_global_metadata[field][i]
                else:
                    channel_global_metadata[field] = original_global_metadata[field]
            channel_global_metadata["PixelsDimensionOrder"] = "ZYX"
            export_global_metadata.append(channel_global_metadata)

    else:
        # Wrapped in list to keep consistent formats and data types
        channels_dataset = [image]
        export_frame_metadata = [original_frame_metadata]
        export_global_metadata = [original_global_metadata]

    # Close PIMS Bioformats reader
    file_image.close()

    return (
        channels_dataset,
        original_global_metadata,
        original_frame_metadata,
        export_global_metadata,
        export_frame_metadata,
    )