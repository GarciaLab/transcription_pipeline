from .nuclear_analysis import segmentation
from .utils import neighborhood_manipulation
from .tracking import track_features, detect_mitosis, stitch_tracks
from .utils import parallel_computing
import warnings
import numpy as np
import pandas as pd
import zarr
from skimage.io import imsave, imread
from pathlib import Path
import pickle


def choose_nuclear_analysis_parameters(
    *,
    nuclear_size,
    sigma_ratio,
    channel_global_metadata,
    division_trigger,
    pos_columns,
    search_range_um,
):
    """
    Chooses reasonable default parameters based on the physical size in each axis of
    the features being segmented in microns and the resolution of the imaging, queried
    from the global metadata corresponding to that image.

    :param nuclear_size: Size of features being segmented in each axis, in microns.
    :type nuclear_size: array-like
    :param float sigma_ratio: Ratio of standard deviations used in bandpass filtering
        using difference of gaussians. Larger ratios widen the bandwidth of the filter.
    :param dict channel_global_metadata: Dictionary of global metadata for the relevant
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param division_trigger: Image feature to use to determine the division frames.
    :type division_trigger: {'num_objects', 'nuclear_fluorescence'}
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param float search_range_um: The maximum distance features in microns can move between
        frames.
    :return: Dictionary of kwarg dicts corresponding to each function in the nuclear
        segmentation and tracking pipeline:

        * `nuclear_analysis.segmentation.denoise_movie`
        * `nuclear_analysis.segmentation.binarize_movie`
        * `nuclear_analysis.segmentation.mark_movie`
        * `nuclear_analysis.segmentation.segment_movie`
        * `nuclear_analysis.segmentation.segmentation_df`
        * `nuclear_analysis.segmentation.link_df`
        * `nuclear_analysis.detect_mitosis.construct_lineage`

    :rtype: dict
    """
    # Query resolution of imaging to translate from physical size to pixels
    mppZ = channel_global_metadata["PixelsPhysicalSizeZ"]
    mppY = channel_global_metadata["PixelsPhysicalSizeY"]
    mppX = channel_global_metadata["PixelsPhysicalSizeX"]

    if mppY != mppX:
        warnings.warn(
            "".join(
                [
                    "Resolution is anisotropic in X and Y, segmentation",
                    " parameters should be handled manually.",
                ]
            )
        )

    nuclear_size_pixels = np.asarray(nuclear_size) / np.array([mppZ, mppY, mppX])

    image_dimensions = np.array(
        [
            channel_global_metadata["PixelsSizeZ"],
            channel_global_metadata["PixelsSizeY"],
            channel_global_metadata["PixelsSizeX"],
        ]
    )

    # Make dictionaries for parameters of `nuclear_analysis` functions.
    denoising_sigma = nuclear_size_pixels / 20
    denoise_params = {"denoising": "gaussian", "denoising_sigma": denoising_sigma}

    opening_footprint_dimensions = np.floor(
        np.maximum(nuclear_size_pixels / 10, 3)
    ).astype(int)
    opening_closing_footprint = neighborhood_manipulation.ellipsoid(
        3, opening_footprint_dimensions[0]
    )

    cc_min_span = nuclear_size_pixels * 2
    # We only check the size in xy of connected components to identify surface noise
    cc_min_span[0] = 0
    background_max_span = image_dimensions
    background_max_span[0] = nuclear_size_pixels[0] / 2
    background_sigma = nuclear_size_pixels * 2
    background_sigma[0] = nuclear_size_pixels[0] / 50
    background_dilation_footprint = neighborhood_manipulation.ellipsoid(
        nuclear_size[1], 3
    )
    binarize_params = {
        "thresholding": "global_otsu",
        "opening_footprint": opening_closing_footprint,
        "closing_footprint": opening_closing_footprint,
        "cc_min_span": cc_min_span,
        "background_max_span": background_max_span,
        "background_sigma": background_sigma,
        "background_threshold_method": "otsu",
        "background_dilation_footprint": background_dilation_footprint,
    }

    # The guidelines for choosing DoG sigmas can be found here:
    # 10.1016/j.jsb.2009.01.004
    k = sigma_ratio
    low_sigma = (
        0.5 * nuclear_size_pixels * np.sqrt(((k**2) - 1) / (2 * (k**2) * np.log(k)))
    )
    high_sigma = low_sigma * k
    # The maximum number of iterations of max-dilations used to find the peaks in the
    # nuclear DoG is set to have a maximum effective footprint of lengthscale twice
    # that of the nuclear diameter to avoid stopping too early before spurious peaks
    # are removed. This corresponds to iterating the minimal-size footprint (3 px)
    # once for every pixel in the lengthcale of a feature since each iteration dilates
    # a maximum by 2 px.
    max_iter = np.ceil(nuclear_size_pixels[1:].max()).astype(int)
    max_footprint = ((1, max_iter), neighborhood_manipulation.ellipsoid(3, 5))
    mark_params = {
        "low_sigma": low_sigma,
        "high_sigma": high_sigma,
        "max_footprint": max_footprint,
        "max_diff": 1,
    }

    volume_nucleus = (
        (4 / 3) * np.pi * np.prod(nuclear_size_pixels / 2)
    )  # Treat nucleus as ellipsoid
    min_size = volume_nucleus / 20
    segment_params = {"watershed_method": "raw", "min_size": min_size}

    # We use the number of nuclei per FOV to determine the nuclear cycle we're in.
    # This requires an estimate of the number of nuclei per square microns:
    num_nuclei_per_um2 = {
        11: (0.0035, 0.0055),
        12: (0.0060, 0.0090),
        13: (0.0100, 0.0175),
        14: (0.0180, 0.0300),
    }
    fov_area = np.prod(image_dimensions[1:]) * mppY * mppX
    num_nuclei_per_fov = {
        key: np.asarray(num_nuclei_per_um2[key]) * fov_area
        for key in num_nuclei_per_um2
    }
    if division_trigger == "num_objects":
        segmentation_df_params = {
            "num_nuclei_per_fov": num_nuclei_per_fov,
            "trigger_property": "num_objects",
            "height": 0.1,
            "distance": 10,
        }
    elif division_trigger == "nuclear_fluorescence":
        # Assume we are going off of loss of nuclear fluorescence during divisions.
        # If we wish to trigger off of loss of exclusion instead (e.g. with MCP-mCherry)
        # this can be edited manually.
        segmentation_df_params = {
            "num_nuclei_per_fov": num_nuclei_per_fov,
            "trigger_property": "nuclear_fluorescence",
            "fluorescence_field": "intensity_mean",
            "invert": True,
            "prominence": 0.5,
        }
    else:
        raise ValueError("`division_trigger` option not recognized.")

    # A search range of ~3.5 um seems to work very well for tracking nuclei in the XY
    # plane. 3D tracking is not as of now handled in the defaults and has to be
    # explicitly passed along with an appropriate search range
    t_column = "frame_reverse"  # Track reversed time to help with large accelerations
    link_df_params = {
        "search_range": search_range_um,
        "memory": 1,
        "pos_columns": pos_columns,
        "t_column": t_column,
        "velocity_predict": True,
        "velocity_averaging": 2,
        "reindex": True,
    }

    # The search range for sibling nuclei can be significantly larger, since only new
    # nuclei are considered as candidate at each linking step
    search_range_mitosis = search_range_um * 2
    construct_lineage_params = {
        "pos_columns": pos_columns,
        "search_range_mitosis": search_range_mitosis,
        "adaptive_stop": 0.05,
        "adaptive_step": 0.99,
        "antiparallel_coordinate": "collision",
        "antiparallel_weight": None,
        "min_track_length": 3,
        "image_dimensions": image_dimensions[1:],
        "exclude_border": 0.02,
        "minimum_age": 4,
    }

    default_params = {
        "denoise_params": denoise_params,
        "binarize_params": binarize_params,
        "mark_params": mark_params,
        "segment_params": segment_params,
        "segmentation_df_params": segmentation_df_params,
        "link_df_params": link_df_params,
        "construct_lineage_params": construct_lineage_params,
    }

    return default_params


# Used for quantification of nuclear protein - this is defined here in global scope
# to help pickling of calculated properties (local objects cannot be pickled).
def intensity_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region], ddof=1)


class Nuclear:
    """
    Runs through the nuclear segmentation and tracking pipeline, using nuclear size
    and scope metadata to come up with reasonable default parameters.

    :param data: Nuclear channel data, in the usual axis ordering ('tzyx').
    :type data: np.ndarray
    :param dict global_metadata: Dictionary of global metadata for the nuclear
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for the nuclear
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param nuclear_size: Size of features being segmented in each axis, in microns.
    :type nuclear_size: array-like
    :param division_trigger: Image feature to use to determine the division frames.
    :type division_trigger: {'num_objects', 'nuclear_fluorescence'}
    :param float sigma_ratio: Ratio of standard deviations used in bandpass filtering
        using difference of gaussians. Larger ratios widen the bandwidth of the filter.
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param float search_range_um: The maximum distance features in microns can move between
        frames.
    :param bool stitch: If `True`, attempts to stitch together filtered tracks by mean
        position and separation in time.
    :param float stitch_max_distance: Maximum distance between mean position of partial tracks
        that still allows for stitching to occur. If `None`, a default of
        0.5*`retrack_search_range_um` is used.
    :param int stitch_max_frame_distance: Maximum number of frames between tracks with no
        points from either tracks that still allows for stitching to occur.
    :param int stitch_frames_mean: Number of frames to average over when estimating the mean
        position of the start and end of candidate tracks to stitch together.
    :param stitch_pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate to use for stitching partial tracks.
    :type stitch_pos_columns: list of DataFrame column names
    :param list series_splits: list of first frame of each series. This is useful
           when stitching together z-coordinates to improve tracking when the z-stack
           has been shifted mid-imaging.
    :param list series_shifts: list of estimated shifts in pixels (sub-pixel
           approximated using centroid of normalized correlation peak) between stacks
           at interface between separate series - this quantifies the shift in the
           z-stack.
    :param bool evaluate: If `True`, each step of the pipeline is fully evaluated
        to produce a Numpy array that gets added as an attribute. Otherwise that
        attribute is set to `None`. The final labelled and tracked array `reordered_labels`
        is always evaluated and kept.
    :param bool keep_futures: If `True`, each step of the pipeline saves the list of
        Futures objects corresponding to its output, adding it as an attribute. Otherwise
        that attribute is deleted and set to `None` once it is no longer needed .The final
        labelled and tracked output `reordered_labels_futures` is always kept.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.

    :ivar default_params: Dictionary of dictionaries, with each subdictionary corresponding
        to the kwargs for one of the functions in the nuclear analysis pipeline, as described
        in the documentation for `choose_nuclear_analysis_parameters`.
    :ivar data_futures: Input nuclear channel data from `data` as a list of scattered
        futures in the Dask Client worker memories.
    :ivar denoised: Input `data`, denoised with the specified method (default is
        Gaussian kernel).
    :ivar denoised_futures: Input `data`, denoised with the specified method (default is
        Gaussian kernel) as a list of scattered futures in the Dask Client worker memories.
    :ivar mask: Boolean mask separating nuclei in foreground from background.
    :ivar mask_futures: Boolean mask separating nuclei in foreground from background as a
        list of scattered futures in the Dask Client worker memories.
    :ivar markers: Boolean array of same shape as `data` with each proposed nucleus marked
        with a single `True` value.
    :ivar markers_futures: Boolean array of same shape as `data` with each proposed
        nucleus marked with a single `True` value, as a list of scattered futures in the
        Dask Client worker memories.
    :ivar labels: Labelled nuclear mask, with each nucleus assigned a unique integer value.
    :ivar labels_futures: Labelled nuclear mask, with each nucleus assigned a unique
        integer value, as a list of scattered futures in the Dask Client worker memories.
    :ivar segmentation_dataframe: pandas DataFrame of frame, label, centroids, and imaging
        time `t_s` for each labelled region in the segmentation mask (along with other
        measurements specified by extra_properties). Also includes column `t_frame` for the
        imaging time in units of z-stack scanning time, and columns `frame_reverse` and
        `t_frame_reverse` with the frame numbers reversed to allow tracking in reverse
        (this performs better on high-acceleration, low-deceleration particles), along
        with the assigned nuclear cycle for the particle as per `assign_nuclear_cycle`.
    :ivar division_frames: Numpy array of frame number (not index - the frames are
        1-indexed as per `trackpy`'s convention) of the detected division windows.
    :ivar nuclear_cycle: Numpy array of nuclear cycle being exited at corresponding
        entry of `division_frames` - this will be one entry larger than `division_frames`
        since we obviously don't see division out of the last cycle observed.
    :ivar tracked_dataframe: `segmentation_dataframe` with an added `particle` column
        assigning an ID to each unique nucleus as tracked by `trackpy` and velocity columns
        for each coordinate in `pos_columns`.
    :ivar mitosis_dataframe: `tracked_dataframe` with IDs in `particle` column reindexed
        so that daughter nuclei both get a fresh ID after a division. A `parent` column
        is also added to describe the nucleus' lineage.
    :ivar reordered_labels: Nuclear segmentation mask, with labels now corresponding to
        the IDs in the `particle` column of `mitosis_dataframe`.
    :ivar reordered_labels_futures: Nuclear segmentation mask, with labels now
        corresponding to the IDs in the `particle` column of `mitosis_dataframe`, as a
        list of scattered futures in the Dask Client worker memories.

    .. note::

        * `nuclear_size` was chosen to be 8.0 :math:`\mu m` in the z-axis and  4.2
          :math:`\mu m` in the x- and y- axes to match the lengthscale of nuclei in
          nc13-nc14 of the early Drosphila embryo, with the width of the bandpass
          filter set by `sigma_ratio` being set to a default 5 to ensure a wide enough
          band that nuclei ranging from nc12 to nc14 are accurately detected.
    """

    def __init__(
        self,
        *,
        data=None,
        global_metadata=None,
        frame_metadata=None,
        client=None,
        nuclear_size=[8.0, 4.2, 4.2],
        division_trigger="num_objects",
        sigma_ratio=5,
        pos_columns=["y", "x"],
        search_range_um=3.5,
        stitch=True,
        stitch_max_distance=None,
        stitch_max_frame_distance=3,
        stitch_frames_mean=4,
        stitch_pos_columns=["y", "x"],
        series_splits=None,
        series_shifts=None,
        evaluate=False,
        keep_futures=False,
    ):
        """
        Constructor method. Instantiates class with no attributes if `data=None`.
        """
        self.labels = None
        self.reordered_labels_futures = None
        self.reordered_labels = None
        self.labels_futures = None
        self.mitosis_dataframe = None
        self.tracked_dataframe = None
        self.markers_futures = None
        self.mask_futures = None
        self.denoised_future = None
        self.data_futures = None
        self.markers = None
        self.mask = None
        if data is not None:
            self.global_metadata = global_metadata
            self.frame_metadata = frame_metadata
            self.nuclear_size = nuclear_size
            self.division_trigger = division_trigger
            self.sigma_ratio = sigma_ratio
            self.pos_columns = pos_columns
            self.search_range_um = search_range_um
            self.stitch = stitch
            self.stitch_pos_columns = stitch_pos_columns
            self.stitch_max_distance = stitch_max_distance
            self.stitch_max_frame_distance = stitch_max_frame_distance
            self.stitch_frames_mean = stitch_frames_mean
            self.series_splits = series_splits
            self.series_shifts = series_shifts
            self.data = data
            self.client = client

            self.default_params = choose_nuclear_analysis_parameters(
                nuclear_size=self.nuclear_size,
                sigma_ratio=self.sigma_ratio,
                channel_global_metadata=self.global_metadata,
                division_trigger=self.division_trigger,
                pos_columns=self.pos_columns,
                search_range_um=self.search_range_um,
            )
            self.evaluate = evaluate
            self.keep_futures = keep_futures

    def track_nuclei(
        self,
        working_memory_mode="zarr",
        working_memory_folder=None,
        monitor_progress=True,
        trackpy_log_path="/tmp/trackpy_log",
        rescale=True,
        only_retrack=False,
    ):
        """
        Runs through the nuclear segmentation and tracking pipeline using the parameters
        instantiated in the constructor method (or any modifications applied to the
        corresponding class attributes), instantiating a class attribute for the output
        (evaluated and Dask Futures objects) at each step if requested.

        :param working_memory_mode: Sets whether the intermediate steps that need to be
            evaluated to construct the nuclear tracking (e.g. construction of a nuclear
            segmentation array) are kept in memory or committed to a `zarr` array. Note
            that using `zarr` as working memory requires the input data to also be a
            `zarr` array, whereby the chunking is inferred from the input data.
        :type working_memory_mode: {"zarr", None}
        :param working_memory_folder: This parameter is required if `working_memory`
            is set to `zarr`, and should be a folder path that points to the location
            where the necessary `zarr` arrays will be stored to disk.
        :type working_memory_folder: {str, `pathlib.Path`, None}
        :param bool monitor_progress: If True, redirects the output of `trackpy`'s
            tracking monitoring to a `tqdm` progress bar.
        :param str trackpy_log_path: Path to log file to redirect trackpy's stdout progress to.
        :param bool rescale: If `True`, rescales particle positions to correspond
            to real space.
        :param bool only_retrack: If `True`, only steps subsequent to tracking are re-run.
        """
        # If `data` is passed as a zarr array, we wrap it as a list of futures
        # that read each chunk - the parallelization is fully determined by the
        # chunking of the data, so memory management can be done by rechunking
        zarr_in_mode = isinstance(self.data, zarr.core.Array)
        zarr_out_mode = (self.client is not None) & (working_memory_mode == "zarr")
        zarr_to_futures = zarr_in_mode & zarr_out_mode

        if zarr_to_futures:
            chunk_boundaries, data = parallel_computing.zarr_to_futures(
                self.data, self.client
            )
        elif zarr_in_mode:
            if working_memory_mode == "zarr":
                raise ValueError(
                    "".join(
                        [
                            "Using `working_memory_mode = 'zarr'` requires",
                            " use of a Dask client.",
                        ]
                    )
                )
            data = self.data[:]
            chunk_boundaries = None
            warnings.warn("Working in local memory.")
        else:
            if working_memory_mode == "zarr":
                raise ValueError(
                    "".join(
                        [
                            "Using `working_memory_mode = 'zarr'` requires",
                            " the data to be fed in as a `zarr` array.",
                        ]
                    )
                )
            chunk_boundaries = None
            data = self.data

        if not only_retrack:
            self.default_params["segmentation_df_params"]["extra_properties"] = (
                "intensity_mean",
            )

            self.default_params["segmentation_df_params"][
                "extra_properties_callable"
            ] = (intensity_stdev,)

            (
                self.denoised,
                self.denoised_futures,
                self.data_futures,
            ) = segmentation.denoise_movie_parallel(
                data,
                client=self.client,
                evaluate=self.evaluate,
                futures_in=True,
                futures_out=True,
                **(self.default_params["denoise_params"]),
            )

            self.mask, self.mask_futures, _ = segmentation.binarize_movie_parallel(
                self.denoised_futures,
                client=self.client,
                evaluate=self.evaluate,
                futures_in=False,
                futures_out=True,
                **(self.default_params["binarize_params"]),
            )

            self.markers, self.markers_futures, _ = segmentation.mark_movie_parallel(
                *(
                    self.data_futures
                ),  # Wrapped in list from previous parallel run, needs unpacking
                self.mask_futures,
                client=self.client,
                evaluate=self.evaluate,
                futures_in=False,
                futures_out=True,
                **(self.default_params["mark_params"]),
            )
            if not self.keep_futures:
                del self.data_futures
                self.data_futures = None

            # If working memory set to "zarr", commit to a zarr array instead of
            # using "evaluate = True", which would pull the whole array into memory
            if zarr_to_futures:
                segment_evaluate = False

                working_memory_path = Path(working_memory_folder)
                results_path = working_memory_path / "nuclear_analysis_results"
                results_path.mkdir(exist_ok=True)

                labels = zarr.creation.zeros_like(
                    self.data,
                    overwrite=True,
                    store=results_path / "segmentation.zarr",
                    dtype=np.uint32,
                )

            else:
                labels = None
                segment_evaluate = True

            self.labels, self.labels_futures, _ = segmentation.segment_movie_parallel(
                self.denoised_futures,
                self.markers_futures,
                self.mask_futures,
                client=self.client,
                evaluate=segment_evaluate,
                futures_in=False,
                futures_out=True,
                **(self.default_params["segment_params"]),
            )

            if zarr_to_futures:
                parallel_computing.futures_to_zarr(
                    self.labels_futures, chunk_boundaries, labels, self.client
                )
                self.labels = labels

                if not self.keep_futures:
                    del self.labels_futures
                    self.labels_futures = None

            if not self.keep_futures:
                del self.denoised_futures
                self.denoised_future = None
                del self.mask_futures
                self.mask_futures = None
                del self.markers_futures
                self.markers_futures = None

            (
                self.segmentation_dataframe,
                self.division_frames,
                self.nuclear_cycle,
            ) = track_features.segmentation_df(
                self.labels,
                self.data,
                self.frame_metadata,
                **(self.default_params["segmentation_df_params"]),
            )

            # Back up pixel-space (unshifted) positions
            pixels_column_names = ["".join([pos, "_pixel"]) for pos in self.pos_columns]
            for i, _ in enumerate(pixels_column_names):
                self.segmentation_dataframe[pixels_column_names[i]] = (
                    self.segmentation_dataframe[self.pos_columns[i]].copy()
                )

            if (self.series_shifts is not None) and ("z" in self.pos_columns):
                # Account for z-stack shift between series
                for i, shift_frame in enumerate(self.series_splits):
                    self.segmentation_dataframe.loc[
                        self.segmentation_dataframe["frame"] > shift_frame, "z"
                    ] = (
                        self.segmentation_dataframe.loc[
                            self.segmentation_dataframe["frame"] > shift_frame, "z"
                        ]
                        - self.series_shifts[i]
                    )

            if rescale:
                # Rescale pixel-space position columns to match real space
                if self.global_metadata is not None:
                    mpp_pos_fields = [
                        "".join(["PixelsPhysicalSize", pos.capitalize()])
                        for pos in self.pos_columns
                    ]
                    mpp_pos = [self.global_metadata[field] for field in mpp_pos_fields]

                    for i, _ in enumerate(self.pos_columns):
                        self.segmentation_dataframe[self.pos_columns[i]] = (
                            self.segmentation_dataframe[self.pos_columns[i]]
                            * mpp_pos[i]
                        )

        self.tracked_dataframe = track_features.link_df(
            self.segmentation_dataframe,
            monitor_progress=monitor_progress,
            trackpy_log_path=trackpy_log_path,
            **(self.default_params["link_df_params"]),
        )

        if self.stitch:
            if self.stitch_max_distance is None:
                self.stitch_max_distance = 0.5 * self.search_range_um

            stitch_tracks.stitch_tracks(
                self.tracked_dataframe,
                self.stitch_pos_columns,
                self.stitch_max_distance,
                self.stitch_max_frame_distance,
                self.stitch_frames_mean,
                inplace=True,
            )

        self.mitosis_dataframe = detect_mitosis.construct_lineage(
            self.tracked_dataframe,
            **(self.default_params["construct_lineage_params"]),
        )

        # Restore dataframe to pixel-space
        pixels_column_names = ["".join([pos, "_pixel"]) for pos in self.pos_columns]
        if (self.series_shifts is not None) and ("z" in self.pos_columns):
            for i, _ in enumerate(self.pos_columns):
                self.mitosis_dataframe[self.pos_columns[i]] = self.mitosis_dataframe[
                    pixels_column_names[i]
                ]
                self.mitosis_dataframe = self.mitosis_dataframe.drop(
                    pixels_column_names[i], axis=1
                )

        # Rename nuclear quantification columns
        self.mitosis_dataframe.rename(
            columns={
                "intensity_mean": "nuclear_intensity_mean",
                "intensity_stdev": "nuclear_intensity_stdev",
            },
            inplace=True,
        )

        if self.client is not None:
            if zarr_in_mode:
                first_last_frames = np.array(chunk_boundaries) + 1
                first_last_frames[:, 1] -= 1
                first_last_frames = first_last_frames.tolist()

                if self.labels_futures is None:
                    _, self.labels_futures = parallel_computing.zarr_to_futures(
                        self.labels, self.client
                    )

            else:
                first_last_frames = None

            if zarr_to_futures:
                reorder_evaluate = False

                working_memory_path = Path(working_memory_folder)
                results_path = working_memory_path / "nuclear_analysis_results"
                results_path.mkdir(exist_ok=True)

                reordered_labels = zarr.creation.zeros_like(
                    self.data,
                    overwrite=True,
                    store=results_path / "reordered_labels.zarr",
                    dtype=np.uint32,
                )

            else:
                reorder_evaluate = True
                reordered_labels = None

            (
                self.reordered_labels,
                self.reordered_labels_futures,
                _,
            ) = track_features.reorder_labels_parallel(
                self.labels_futures,
                self.mitosis_dataframe,
                client=self.client,
                first_last_frames=first_last_frames,
                evaluate=reorder_evaluate,
                futures_in=False,
                futures_out=True,
            )

            if zarr_to_futures:
                parallel_computing.futures_to_zarr(
                    self.reordered_labels_futures,
                    chunk_boundaries,
                    reordered_labels,
                    self.client,
                )
                self.reordered_labels = reordered_labels

            if not self.keep_futures:
                del self.labels_futures
                self.labels_futures = None

                del self.reordered_labels_futures
                self.reordered_labels_futures = None

        else:
            self.reordered_labels = track_features.reorder_labels(
                self.labels, self.mitosis_dataframe
            )
            self.reordered_labels_futures = None

        if not self.evaluate:
            del self.labels
            self.labels = None

    def save_results(self, *, name_folder, save_array_as="zarr", save_all=False):
        """
        Saves results of nuclear segmentation and tracking to disk as HDF5, with the
        labelled segmentation mask saved as zarr or TIFF as specified.

        :param str name_folder: Name of folder to create `nuclear_analysis_results`
            subdirectory in, in which analysis results are stored.
        :param save_array_as: Format to save reordered nuclear labels in:
        :type save_array_as: {"zarr", "tiff"}
        :param bool save_all: If true, saves a tiff file for each intermediate step
            of the nuclear analysis pipeline
        """
        # Make `nuclear_analysis_results` directory if it doesn't exist
        name_path = Path(name_folder)
        results_path = name_path / "nuclear_analysis_results"
        results_path.mkdir(exist_ok=True)

        # Save movies, saving intermediate steps if requested
        save_movies = {"reordered_labels": self.reordered_labels}

        if save_all:
            save_movies["denoised"] = self.denoised
            save_movies["binarized"] = self.mask
            save_movies["watershed_markers"] = self.markers

        file_not_found = "".join(
            [
                " could not be found, check to make",
                " sure the `evaluate` kwarg was used when running the",
                " `track_nuclei` method so that the",
                " intermediate steps are not deleted to save",
                " memory.",
            ]
        )

        if save_array_as == "zarr":
            for movie in save_movies:
                if save_movies[movie] is not None:
                    save_name = "".join([movie, ".zarr"])
                    save_path = results_path / save_name
                    zarr.save_array(str(save_path), save_movies[movie])
                else:
                    warnings.warn("".join([movie, file_not_found]))

        elif save_array_as == "tiff":
            for movie in save_movies:
                if save_movies[movie] is not None:
                    save_name = "".join([movie, ".tiff"])
                    save_path = results_path / save_name
                    imsave(save_path, save_movies[movie], plugin="tifffile")
                else:
                    warnings.warn("".join([movie, file_not_found]))

        elif save_array_as is not None:
            raise ValueError("`save_array_as` option not recognized.")

        # Save nuclear dataframes
        mitosis_dataframe_path = results_path / "mitosis_dataframe.pkl"
        self.mitosis_dataframe.to_pickle(mitosis_dataframe_path, compression=None)
        if save_all:
            segmentation_dataframe_path = results_path / "segmentation_dataframe.pkl"
            self.segmentation_dataframe.to_pickle(
                segmentation_dataframe_path, compression=None
            )
            tracked_dataframe_path = results_path / "tracked_dataframe.pkl"
            self.tracked_dataframe.to_pickle(tracked_dataframe_path, compression=None)

        # Keep a copy of the parameters used
        nuclear_analysis_param_path = results_path / "nuclear_analysis_parameters.pkl"
        with open(nuclear_analysis_param_path, "wb") as f:
            pickle.dump(self.default_params, f)

    def read_results(self, *, name_folder, import_all=False):
        """
        Imports results from a saved run of `track_nuclei` into the corresponding
        class attributes.

        :param str name_folder: Name of folder where `nuclear_analysis_results`
            subdirectory resides.
        :param bool import_all: If `True`, will attempt to also import results of
            intermediate steps of nuclear analysis if they have been saved to disk,
            showing a warning if a corresponding file cannot be found in the specified
            directory.
        """
        name_path = Path(name_folder)
        results_path = name_path / "nuclear_analysis_results"

        # Import arrays
        import_arrays = ["reordered_labels"]
        if import_all:
            import_arrays.extend(
                [
                    "segmentation",
                    "watershed_markers",
                    "denoised",
                    "binarized",
                ]
            )

        for array in import_arrays:
            # Try importing as zarr first - if there is no zarr array, this will
            # return `None`
            zarr_path = results_path / "".join([array, ".zarr"])
            setattr(self, array, zarr.open(str(zarr_path)))

            if getattr(self, array) is None:
                tiff_path = results_path / "".join([array, ".tiff"])
                try:
                    setattr(self, array, imread(tiff_path, plugin="tifffile"))
                except FileNotFoundError:
                    warnings.warn("".join([array, " not found, keeping as `None`."]))

        # Import dataframes
        import_dataframes = ["mitosis_dataframe"]
        if import_all:
            import_dataframes.extend(["segmentation_dataframe", "tracked_dataframe"])
        for dataframe in import_dataframes:
            dataframe_path = results_path / "".join([dataframe, ".pkl"])
            try:
                setattr(
                    self, dataframe, pd.read_pickle(dataframe_path, compression=None)
                )
            except FileNotFoundError:
                setattr(self, dataframe, None)
                warnings.warn("".join([dataframe, " not found, keeping as `None`."]))

        # Import saved parameters
        with open(results_path / "nuclear_analysis_parameters.pkl", "rb") as f:
            self.default_params = pickle.load(f)
