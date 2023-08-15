from .nuclear_analysis import segmentation
from .tracking import track_features, detect_mitosis
from .preprocessing import import_data
import warnings
import numpy as np
import pandas as pd
import zarr
from skimage.io import imsave, imread
from pathlib import Path
import pickle
import glob


def choose_nuclear_analysis_parameters(
    nuclear_size, sigma_ratio, channel_global_metadata
):
    """
    Chooses reasonable default parameters based on the physical size in each axis of
    the features being segmented in microns and the resolution of the imaging, queried
    from the global metadata corresponding to that image.

    :param nuclear_size: Size of features being segmented in each axis, in microns.
    :type nuclear_size: array-like
    :param float sigma_ratio: Ratio of stndard deviations used in bandpass filtering
        using difference of gaussians. Larger ratios widen the bandwidth of the filter.
    :param dict channel_global_metadata: Dictionary of global metadata for the relevant
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :return: Dictionary of kwarg dicts corresponding to each function in the nuclear
        segmentation and tracking pipeline:
        *`nuclear_analysis.segmentation.denoise_movie`
        *`nuclear_analysis.segmentation.binarize_movie`
        *`nuclear_analysis.segmentation.mark_movie`
        *`nuclear_analysis.segmentation.segment_movie`
        *`nuclear_analysis.segmentation.segmentation_df`
        *`nuclear_analysis.segmentation.link_df`
        *`nuclear_analysis.detect_mitosis.construct_lineage`
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
        np.maximum(nuclear_size_pixels / 2, 3)
    ).astype(int)
    opening_footprint = segmentation.ellipsoid(3, opening_footprint_dimensions[0])
    closing_footprint_dimensions = np.floor(
        np.maximum(nuclear_size_pixels / 10, 3)
    ).astype(int)
    closing_footprint = segmentation.ellipsoid(
        (closing_footprint_dimensions[1:]).max(), closing_footprint_dimensions[0]
    )
    cc_min_span = nuclear_size_pixels * 2
    # We only check the size in xy of connected components to identify surface noise
    cc_min_span[0] = 0
    background_max_span = image_dimensions
    background_max_span[0] = nuclear_size_pixels[0] / 2
    background_sigma = nuclear_size_pixels * 2
    background_sigma[0] = nuclear_size_pixels[0] / 50
    background_dilation_footprint = segmentation.ellipsoid(nuclear_size[1], 3)
    binarize_params = {
        "thresholding": "global_otsu",
        "opening_footprint": opening_footprint,
        "closing_footprint": closing_footprint,
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
    max_footprint = ((1, max_iter), segmentation.ellipsoid(3, 5))
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
    num_nuclei_per_um2 = {12: (0.005, 0.009), 13: (0.012, 0.018), 14: (0.024, 0.040)}
    fov_area = np.prod(image_dimensions[1:]) * mppY * mppX
    num_nuclei_per_fov = {
        key: np.asarray(num_nuclei_per_um2[key]) * fov_area
        for key in num_nuclei_per_um2
    }
    segmentation_df_params = {
        "num_nuclei_per_fov": num_nuclei_per_fov,
        "division_peak_height": 0.1,
        "min_time_between_divisions": 10,
    }

    # A search range of ~3.5 um seems to work very well for tracking nuclei in the XY
    # plane. 3D tracking is not as of now handled in the defaults and has to be
    # explicitly passes along with an appropriate search range
    search_range = 3.5 / mppY
    pos_columns = ["y", "x"]  # Tracking only in XY in case z-stack was adjusted
    t_column = "frame_reverse"  # Track reversed time to help with large accelerations
    link_df_params = {
        "search_range": search_range,
        "memory": 1,
        "pos_columns": pos_columns,
        "t_column": t_column,
        "velocity_predict": True,
        "velocity_averaging": 2,
        "reindex": True,
    }

    # The search range for sibling nuclei can be significantly larger, since only new
    # nuclei are considered as candidate at each linking step
    search_range_mitosis = search_range * 2
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


class Nuclear:
    """
    Runs through the nuclear segmentation and tracking pipeline, using nuclear size
    and scope metadata to come up with reasonable default parameters.

    :param data: Nuclear channel data, in the usual axis ordering ('tzyx').
    :type data: Numpy array.
    :param dict global_metadata: Dictionary of global metadata for the nuclear
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for the nuclear
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param nuclear_size: Size of features being segmented in each axis, in microns.
    :type nuclear_size: array-like
    :param float sigma_ratio: Ratio of stndard deviations used in bandpass filtering
        using difference of gaussians. Larger ratios widen the bandwidth of the filter.
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
        *`nuclear_size` was chosen to be $8.0 \ \mu m$ in the z-axis and $4.2 \ \mu m$ in
        the x- and y- axes to match the lengthscale of nuclei in nc13-nc14 of the early
        Drosphila embryo, with the width of the bandpass filter set by `sigma_ratio` being
        set to a default 5 to ensure a wide enough band that nuclei ranging from nc12 to
        nc14 are accurately detected.
    """

    def __init__(
        self,
        *,
        data=None,
        global_metadata=None,
        frame_metadata=None,
        client=None,
        nuclear_size=[8.0, 4.2, 4.2],
        sigma_ratio=5,
        evaluate=False,
        keep_futures=False,
    ):
        """
        Constructor method. Instantiates class with no attributes if `data=None`.
        """
        if data is not None:
            self.global_metadata = global_metadata
            self.frame_metadata = frame_metadata
            self.nuclear_size = nuclear_size
            self.sigma_ratio = sigma_ratio
            self.data = data
            self.client = client

            self.default_params = choose_nuclear_analysis_parameters(
                self.nuclear_size, self.sigma_ratio, self.global_metadata
            )
            self.evaluate = evaluate
            self.keep_futures = keep_futures

    def track_nuclei(self):
        """
        Runs through the nuclear segmentation and tracking pipeline using the parameters
        instantiated in the constructor method (or any modifications applied to the
        corresponding class attributes), instantiating a class attribute for the output
        (evaluated and Dask Futures objects) at each step if requested.
        """
        (
            self.denoised,
            self.denoised_futures,
            self.data_futures,
        ) = segmentation.denoise_movie_parallel(
            self.data,
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

        self.labels, self.labels_futures, _ = segmentation.segment_movie_parallel(
            self.denoised_futures,
            self.markers_futures,
            self.mask_futures,
            client=self.client,
            evaluate=True,
            futures_in=False,
            futures_out=True,
            **(self.default_params["segment_params"]),
        )
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

        if not self.evaluate:
            del self.labels
            self.labels = None

        self.tracked_dataframe = track_features.link_df(
            self.segmentation_dataframe,
            **(self.default_params["link_df_params"]),
        )

        self.mitosis_dataframe = detect_mitosis.construct_lineage(
            self.tracked_dataframe,
            **(self.default_params["construct_lineage_params"]),
        )

        (
            self.reordered_labels,
            self.reordered_labels_futures,
            _,
        ) = track_features.reorder_labels_parallel(
            self.labels_futures,
            self.mitosis_dataframe,
            client=self.client,
            evaluate=True,
            futures_in=False,
            futures_out=True,
        )
        if not self.keep_futures:
            del self.labels_futures
            self.labels_futures = None

    def save_results(self, *, name_folder, save_array_as="zarr", save_all=False):
        """
        Saves results of nuclear segmentation and tracking to disk as HDF5, with the
        labelled segmentation mask saved as zarr or TIFF as specified.

        :param str file_name: Name to save reordered (post-tracking) labelled nuclear
            mask under.
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
        save_movies = {"reordered_nuclear_labels": self.reordered_labels}

        if save_all:
            save_movies["denoised_nuclear_channel"] = self.denoised
            save_movies["binarized_nuclear_channel"] = self.mask
            save_movies["nuclear_watershed_markers"] = self.markers

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
                    zarr.save_array(save_path, save_movies[movie])
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
        import_arrays = ["reordered_nuclear_labels"]
        if import_all:
            import_arrays.extend(
                [
                    "nuclear_watershed_markers",
                    "denoised_nuclear_channel",
                    "binarized_nuclear_channel",
                ]
            )

        for array in import_arrays:
            # Try importing as zarr first - if there is no zarr array, this will
            # return `None`
            zarr_path = results_path / "".join([array, ".zarr"])
            setattr(self, array, zarr.load(zarr_path))

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