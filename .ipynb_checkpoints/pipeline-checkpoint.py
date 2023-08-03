from nuclear_analysis import segmentation
from tracking import track_features, detect_mitosis
from spot_analysis import detection, fitting, track_filtering
from scipy.optimize import fsolve
import warnings
import numpy as np


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

    # A search range of ~4.2 um seems to work very well for tracking nuclei in the XY
    # plane. 3D tracking is not as of now handled in the defaults and has to be
    # explicitly passes along with an appropriate search range
    search_range = 4.2 / mppY
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
        data,
        global_metadata,
        frame_metadata,
        client,
        nuclear_size=[8.0, 4.2, 4.2],
        sigma_ratio=5,
        evaluate=False,
        keep_futures=False,
    ):
        """
        Constructor method.
        """
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


def _solve_for_sigma(fwhm, sigma_ratio, ndim):
    """
    Finds the standard deviations that construct a DoG filter with specified full
    width at half maximum `fwhm`.
    """

    # We can find the sigmas that ensure a specified FWHM of the DoG by setting the
    # expression for the DoG function to half of its value at the origin, setting the
    # spatial coordinate to be the half the FWHM, and solving for sigma.
    def fwhm_eqn(sigma_1):
        dog = (
            (1 / (sigma_1**ndim)) * np.exp(-((fwhm / 2) ** 2) / (2 * (sigma_1**2)))
            - (1 / ((sigma_ratio * sigma_1) ** ndim))
            * np.exp(-((fwhm / 2) ** 2) / (2 * ((sigma_ratio * sigma_1) ** 2)))
            - 0.5 * ((1 / (sigma_1**ndim)) - (1 / ((sigma_ratio * sigma_1) ** ndim)))
        )
        return dog

    low_sigma = fsolve(fwhm_eqn, fwhm / 2)
    high_sigma = low_sigma * sigma_ratio

    return low_sigma, high_sigma


def choose_spot_analysis_parameters(
    channel_global_metadata,
    spot_sigmas,
    spot_sigma_xy_bounds,
    spot_sigma_z_bounds,
    extract_sigma_multiple,
    dog_sigma_ratio,
    keep_bandpass,
):
    """
    Chooses reasonable default parameters based on provided physical scale in microns
    of the standard deviations in each coordinate axis of diffraction-limited spots.
    This is translated to pixel-space by querying resolution metadata from the global
    metadata corresponding to that image.

    :param dict channel_global_metadata: Dictionary of global metadata for the relevant
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param spot_sigmas: Standard deviations in each coordinate axis of diffraction-
        limited spots. This is best estimated from preliminary data (for instance,
        by choosing a ballpark estimate and running a first pass of the analysis and
        observing the resultant histograms for `sigma_x_y` and `sigma_z`).
    :type spot_sigmas: array-like
    :param spot_sigma_xy_bounds: 2-iterable of lower- and upper-bound on acceptable
        standard deviation in microsn in x- and y-axes in order for a spot to beconsidered
        in downstream analysis. Like `spot_sigmas`, this is best estimated experimentally
        but a ballpark can be estimates from the theoretical PSF of the microscope for a
        first pass.
    :type spot_sigma_xy_bounds: Iterable
    :param spot_sigma_z_bounds: 2-iterable of lower- and upper-bound on acceptable
        standard deviation in microns in z-axis in order for a spot to be considered in
        downstream analysis. Like `spot_sigmas`, this is best estimated experimentally but
        a ballpark can be estimates from the theoretical PSF of the microscope for a
        first pass.
    :type spot_sigma_z_bounds: Iterable
    :param extract_sigma_multiple: Multiple of the proposed `spot_sigmas` in each axis
        used to set the dimensions of the volume that gets extracted out of the spot data
        array into `spot_dataframe` for Gaussian fitting.
    :type extract_sigma_multiple: Array-like
    :param float dog_sigma_ratio: Ratio of standard deviations of Difference of Gaussians
        filter used to preprocess the data.
    :param bool keep_bandpass: If `True`, keeps a copy of the bandpass-filtered image
        in memory.
    :return: Dictionary of kwarg dicts corresponding to each function in the spot
        segmentation and tracking pipeline:
        *`detection.detect_and_gather_spots`
        *`fitting.add_fits_spots_dataframe_parallel`
        *`track_filtering.track_and_filter_spots`
        *`track_features.reorder_labels_parallel`
    :rtype: dict
    """
    spot_sigmas = np.asarray(spot_sigmas)

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

    spot_sigmas_pixels = spot_sigmas / np.array([mppZ, mppY, mppX])

    # To determine the best sigmas for the DoG filter, we match the FWHM of the DoG
    # with the FWHM of the Gaussian spot it is meant to detect
    spot_fwhm = spot_sigmas_pixels * 2 * np.sqrt(2 * np.log(2))
    dog_sigmas = _solve_for_sigma(spot_fwhm, dog_sigma_ratio, spot_sigmas.size)

    detect_and_gather_spots_params = {
        "low_sigma": dog_sigmas[0],
        "high_sigma": dog_sigmas[1],
        "threshold": "triangle",
        "min_size": 4,
        "connectivity": 1,
        "span": spot_sigmas_pixels * np.asarray(extract_sigma_multiple),
        "pos_columns": ["z", "y", "x"],
        "return_bandpass": keep_bandpass,
        "return_spot_mask": True,
        "drop_reverse_time": True,
    }

    add_fits_spots_dataframe_parallel_params = {
        "sigma_x_y_guess": spot_sigmas_pixels[1],
        "sigma_z_guess": spot_sigmas_pixels[0],
        "amplitude_guess": None,
        "offset_guess": None,
        "method": "trf",
        "inplace": True,
    }

    # A search range of ~4.2 um seems to work very well for tracking nuclei in the XY
    # plane. 3D tracking is not as of now handled in the defaults and has to be
    # explicitly passes along with an appropriate search range. We also use a default
    # `memory` parameter of 4 with `trackpy` (nuclear tracking uses a value of 1)
    # since spots move out of focus more easily.
    track_and_filter_spots_params = {
        "sigma_x_y_bounds": np.asarray(spot_sigma_xy_bounds) / mppY,
        "sigma_z_bounds": np.asarray(spot_sigma_z_bounds) / mppZ,
        "expand_distance": 2,
        "search_range": 4.2 / mppY,
        "memory": 2,
        "pos_columns": ["y", "x"],
        "t_column": "frame_reverse",
        "velocity_predict": True,
        "velocity_averaging": None,
        "min_track_length": 5,  # Lax minimum tracked timepoints to filter spurious tracks
        "choose_by": "amplitude",
        "min_or_max": "maximize",
    }

    default_params = {
        "detect_and_gather_spots_params": detect_and_gather_spots_params,
        "add_fits_spots_dataframe_parallel_params": add_fits_spots_dataframe_parallel_params,
        "track_and_filter_spots_params": track_and_filter_spots_params,
    }

    return default_params


class Spot:
    """
    Runs through the spot segmentation, fitting and tracking pipeline, using proposed
    spot sigmas (usually experimentally determined on a preliminary dataset) to come
    up with reasonable default parameters.

    :param data: Spot channel data, in the usual axis ordering ('tzyx').
    :type data: Numpy array.
    :param dict global_metadata: Dictionary of global metadata for the spot
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param dict frame_metadata: Dictionary of frame-by-frame metadata for the spot
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param nuclear_labels: Labelled nuclear mask, with each nucleus assigned a unique
        integer value. Can also be passed as a list of futures corresponding to the
        labelled nuclear mask in Dask worker memories. Setting to `None` results in
        independent tracking and fitting of the spots, without the filtering steps
        that would require a nuclear mask.
    :type nuclear_labels: Numpy array, list of Dask Fututes objects, or `None`
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param spot_sigmas: Standard deviations in each coordinate axis of diffraction-
        limited spots. This is best estimated from preliminary data (for instance,
        by choosing a ballpark estimate and running a first pass of the analysis and
        observing the resultant histograms for `sigma_x_y` and `sigma_z`).
    :type spot_sigmas: array-like
    :param spot_sigma_xy_bounds: 2-iterable of lower- and upper-bound on acceptable
        standard deviation in microsn in x- and y-axes in order for a spot to beconsidered
        in downstream analysis. Like `spot_sigmas`, this is best estimated experimentally
        but a ballpark can be estimates from the theoretical PSF of the microscope for a
        first pass.
    :type spot_sigma_xy_bounds: Iterable
    :param spot_sigma_z_bounds: 2-iterable of lower- and upper-bound on acceptable
        standard deviation in microns in z-axis in order for a spot to be considered in
        downstream analysis. Like `spot_sigmas`, this is best estimated experimentally but
        a ballpark can be estimates from the theoretical PSF of the microscope for a
        first pass.
    :type spot_sigma_z_bounds: Iterable
    :param extract_sigma_multiple: Multiple of the proposed `spot_sigmas` in each axis
        used to set the dimensions of the volume that gets extracted out of the spot data
        array into `spot_dataframe` for Gaussian fitting.
    :type extract_sigma_multiple: Array-like
    :param float dog_sigma_ratio: Ratio of standard deviations of Difference of Gaussians
        filter used to preprocess the data.
    :param bool evaluate: If `True`, fully evaluates tracked and reordered spot labels
        as a Numpy array after gathering Futures from worker memories.
    :param bool keep_futures: If `True`, keeps a pointer to Futures objects in worker
        memories corresponding to tracked and reordered spot labels.
    :param bool keep_bandpass: If `True`, keeps a copy of the bandpass-filtered image
        in memory. This is kept as `dtype=np.float64`, and on machines lacking in memory
        can cause the Python kernel to crash for larger datasets.

    :ivar default_params: Dictionary of dictionaries, with each subdictionary corresponding
        to the kwargs for one of the functions in the spot analysis pipeline, as described
        in the documentation for `choose_spot_analysis_parameters`.
    :ivar spot_dataframe: pandas DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`, along with
        information added by subsequent Gaussian fitting and filtering steps - for details,
        see documentation for `detection.detect_and_gather_spots`,
        `fitting.add_fits_spots_dataframe_parallel`, and
        `track_filtering.track_and_filter_spots`.
    :ivar spot_mask: Boolean mask separating proposed spots in foreground from background.
    :ivar bandpassed_movie: Input spot `data` after bandpass filtering with a Difference
        of Gaussians.
    :ivar reordered_spot_labels: Spot segmentation mask, with labels now corresponding to
        the IDs in the `particle` column of `spot_dataframe`. If available, the IDs are
        transferred over from provided `nuclear_labels`.
    :ivar reordered_spot_labels_futures: Spot segmentation mask, with labels now
        corresponding to the IDs in the `particle` column of `spot_dataframe`, as a list
        of scattered futures in the Dask Client worker memeories. If available, the IDs are
        transferred over from provided `nuclear_labels`.

    .. note::
        *The default $\sigma_z = 0.43 \ \mu m$, $\sigma_{x, y} = 0.21 \ \mu m$ along with the
        corresponding bounds on the standard deviations were chosen empirically by running
        the analysis on a preliminary dataset.
        *`dog_sigma_ratio = 1.6` was chosen to approximate the Laplacian of Gaussians while
        maintaining good response, as per https://doi.org/10.1098/rspb.1980.0020.
        *With the ratio of the standard deviations of the Difference of Gaussians filter
        fixed, the stndard deviations were chosen so that the full width at half-maximum
        of the resultant filter matched that of a Gaussian with standard deviations
        given by `spot_sigmas` so as to maximize the response to spots of the correct scale.
    """

    def __init__(
        self,
        *,
        data,
        global_metadata,
        frame_metadata,
        nuclear_labels,
        client,
        spot_sigmas=[0.43, 0.21, 0.21],
        spot_sigma_x_y_bounds=(0.052, 0.52),
        spot_sigma_z_bounds=(0.16, 1),
        extract_sigma_multiple=[6, 10, 10],
        dog_sigma_ratio=1.6,
        evaluate=True,
        keep_futures=True,
        keep_bandpass=True,
    ):
        """
        Constructor method.
        """
        self.data = data
        self.global_metadata = global_metadata
        self.frame_metadata = frame_metadata
        self.client = client
        self.spot_sigmas = spot_sigmas
        self.spot_sigma_x_y_bounds = spot_sigma_x_y_bounds
        self.spot_sigma_z_bounds = spot_sigma_z_bounds
        self.extract_sigma_multiple = extract_sigma_multiple
        self.dog_sigma_ratio = dog_sigma_ratio
        self.nuclear_labels = nuclear_labels
        self.keep_bandpass = keep_bandpass

        self.default_params = choose_spot_analysis_parameters(
            self.global_metadata,
            self.spot_sigmas,
            self.spot_sigma_x_y_bounds,
            self.spot_sigma_z_bounds,
            self.extract_sigma_multiple,
            self.dog_sigma_ratio,
            self.keep_bandpass,
        )

        self.evaluate = evaluate
        self.keep_futures = keep_futures

    def extract_spot_traces(self):
        """
        Runs through the spot segmentation, tracking, fitting and quantification
        pipeline using the parameters instantiated in the constructor method (or any
        modifications applied to the corresponding class attributes), instantiating a
        class attribute for the output (evaluated and Dask Futures objects) at each
        step if requested.
        """
        (
            self.spot_dataframe,
            self.spot_mask,
            self.bandpassed_movie,
        ) = detection.detect_and_gather_spots(
            self.data,
            frame_metadata=self.frame_metadata,
            client=self.client,
            **(self.default_params["detect_and_gather_spots_params"]),
        )

        fitting.add_fits_spots_dataframe_parallel(
            self.spot_dataframe,
            client=self.client,
            **(self.default_params["add_fits_spots_dataframe_parallel_params"]),
        )

        track_filtering.track_and_filter_spots(
            self.spot_dataframe,
            nuclear_labels=self.nuclear_labels,
            client=self.client,
            **(self.default_params["track_and_filter_spots_params"]),
        )

        (
            self.reordered_spot_labels,
            self.reordered_spot_labels_futures,
            futures_in,
        ) = track_features.reorder_labels_parallel(
            self.spot_mask,
            self.spot_dataframe,
            client=self.client,
            futures_in=False,
            futures_out=self.keep_futures,
            evaluate=self.evaluate,
        )
        del futures_in