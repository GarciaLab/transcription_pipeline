from .tracking import track_features, stitch_tracks
from .spot_analysis import detection, fitting, track_filtering
from .utils import parallel_computing
from scipy.optimize import fsolve
import warnings
import numpy as np
import pandas as pd
import zarr
from skimage.io import imsave, imread
from pathlib import Path
import pickle
import glob
from functools import partial


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


def _transfer_labels(dataframe_future, labels, params, pos_columns):
    """Helper function to transfer labels inside the worker
    memory."""
    # Pull only required frames from labels into memory
    initial_frame = dataframe_future["frame"].min() - 1
    final_frame = dataframe_future["frame"].max()
    labels_subarray = labels[initial_frame:final_frame]

    dataframe = dataframe_future.copy()

    track_filtering.transfer_nuclear_labels(
        dataframe,
        labels_subarray,
        expand_distance=params["track_and_filter_spots_params"]["expand_distance"],
        pos_columns=pos_columns,
        client=None,
    )
    return dataframe


def choose_spot_analysis_parameters(
    *,
    channel_global_metadata,
    dog_sigma_ratio,
    spot_sigmas,
    spot_sigma_x_y_bounds,
    spot_sigma_z_bounds,
    extract_sigma_multiple,
    expand_distance,
    search_range_um,
    memory,
    pos_columns,
    velocity_averaging,
    min_track_length,
    filter_multiple,
    retrack_pos_columns,
    retrack_search_range_um,
    retrack_memory,
    retrack_min_track_length,
    retrack_by_intensity,
    retrack_intensity_normalize_quantile,
    num_bootstraps,
    integrate_sigma_multiple,
    keep_bandpass,
    keep_spot_labels,
):
    """
    Chooses reasonable default parameters based on provided physical scale in microns
    of the standard deviations in each coordinate axis of diffraction-limited spots.
    This is translated to pixel-space by querying resolution metadata from the global
    metadata corresponding to that image.

    :param dict channel_global_metadata: Dictionary of global metadata for the relevant
        channel, as output by `preprocessing.import_data.import_save_dataset`.
    :param float dog_sigma_ratio: Ratio of standard deviations of Difference of Gaussians
        filter used to preprocess the data.
    :param spot_sigmas: Standard deviations in each coordinate axis of diffraction-
        limited spots. This is best estimated from preliminary data (for instance,
        by choosing a ballpark estimate and running a first pass of the analysis and
        observing the resultant histograms for `sigma_x_y` and `sigma_z`).
    :type spot_sigmas: array-like
    :param spot_sigma_x_y_bounds: 2-iterable of lower- and upper-bound on acceptable
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
    :param int expand_distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :param float search_range_um: The maximum distance features in microns can move between
        frames.
    :param int memory: The maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle.
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param int averaging: Number of frames to average velocity over.
    :param int min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis.
    :param bool filter_multiple: Decide whether or not to enforce a single-spot
        limit for each nucleus.
    :param retrack_pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type retrack_pos_columns: list of DataFrame column names
    :param float retrack_search_range_um: The maximum distance features in microns can move
        between frames during the second tracking (after filtering by track length).
    :param int retrack_memory: The maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle during the second tracking
        (after filtering by track length
    :param int retrack_min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis during the second tracking
        (after filtering by track length
    :param bool retrack_by_intensity: If `True`, this will attempt to use a preliminary
        spot tracking to use the variation of intensity across traces to help spot
        tracking (i.e. spots with wild jumps in intensity are less likely to be linked
        across frames). See documentation for `normalized_variation_intensity` for
        details.
    :param float retrack_intensity_normalize_quantile: Target value of .84-quantile of
        successive differences in intensity across traces after normalization. This can
        essentially be used to set the "exchange rate" between spatial proximity and
        similarity in intensity (i.e. when tracking, differing in intensity by the
        .84-quantile is penalized as much as being separated by `normalize_quantile_to` from
        a candidate point in the next frame) - the higher the value use, the less tolerant
        tracking is of large variations in intensity.
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :param integrate_sigma_multiple: Multiple of the proposed `spot_sigmas` in all axes
        used to set the dimensions of the ellipsoid that gets integrated over for spot
        quantification.
    :param bool keep_bandpass: If `True`, keeps a copy of the bandpass-filtered image
        in memory.
    :param bool keep_spot_labels: If `True`, keeps a copy of the spot mask after thresholding
        and labeling but before filtering in memory.
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
        "return_spot_labels": keep_spot_labels,
        "drop_reverse_time": True,
    }

    add_fits_spots_dataframe_parallel_params = {
        "sigma_x_y_guess": spot_sigmas_pixels[1],
        "sigma_z_guess": spot_sigmas_pixels[0],
        "amplitude_guess": None,
        "offset_guess": None,
        "method": "trf",
        "inplace": False,
    }

    add_neighborhood_intensity_spot_dataframe_parallel_params = {
        "mppZ": mppZ,
        "mppYX": mppY,
        "ball_diameter_um": spot_sigmas[1] * integrate_sigma_multiple,
        "shell_width_um": mppY * 1.1,  # Ensures that a continuous ring is extracted
        "aspect_ratio": spot_sigmas[1] / spot_sigmas[0],
        "num_bootstraps": num_bootstraps,
        "inplace": False,
    }

    # A search range of ~4.2 um seems to work very well for tracking nuclei in the XY
    # plane. 3D tracking is not as of now handled in the defaults and has to be
    # explicitly passes along with an appropriate search range. We also use a default
    # `memory` parameter of 4 with `trackpy` (nuclear tracking uses a value of 1)
    # since spots move out of focus more easily.

    track_and_filter_spots_params = {
        "sigma_x_y_bounds": np.asarray(spot_sigma_x_y_bounds) / mppY,
        "sigma_z_bounds": np.asarray(spot_sigma_z_bounds) / mppZ,
        "expand_distance": expand_distance,
        "search_range": search_range_um,
        "memory": memory,
        "pos_columns": pos_columns,
        "t_column": "frame_reverse",
        "velocity_predict": True,
        "velocity_averaging": velocity_averaging,
        "min_track_length": min_track_length,
        "filter_multiple": filter_multiple,
        "choose_by": "intensity_from_neighborhood",
        "min_or_max": "maximize",
        "filter_negative": True,
        "quantification": "intensity_from_neighborhood",
        "track_by_intensity": False,
    }

    retrack_spots_params = {
        "sigma_x_y_bounds": np.asarray(spot_sigma_x_y_bounds) / mppY,
        "sigma_z_bounds": np.asarray(spot_sigma_z_bounds) / mppZ,
        "expand_distance": expand_distance,
        "search_range": retrack_search_range_um,
        "memory": retrack_memory,
        "pos_columns": retrack_pos_columns,
        "t_column": "frame_reverse",
        "velocity_predict": True,
        "velocity_averaging": velocity_averaging,
        "min_track_length": retrack_min_track_length,
        "filter_multiple": False,
        "filter_negative": False,
        "track_by_intensity": retrack_by_intensity,
        "normalize_quantile_to": retrack_intensity_normalize_quantile,
        "min_track_length_intensity": retrack_min_track_length,
    }

    default_params = {
        "detect_and_gather_spots_params": detect_and_gather_spots_params,
        "add_fits_spots_dataframe_parallel_params": add_fits_spots_dataframe_parallel_params,
        "add_neighborhood_intensity_spot_dataframe_parallel_params": add_neighborhood_intensity_spot_dataframe_parallel_params,
        "track_and_filter_spots_params": track_and_filter_spots_params,
        "retrack_spots_params": retrack_spots_params,
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
    :param labels: Labelled mask, with each label assigned a unique integer value and
        containing a single spot. Can also be passed as a list of futures corresponding to
        the labelled mask in Dask worker memories. Setting to `None` results in
        independent tracking and fitting of the spots, without the filtering steps
        that would require a nuclear mask.
    :type labels: Numpy array, list of Dask Fututes objects, or `None`
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param spot_sigmas: Standard deviations in each coordinate axis of diffraction-
        limited spots. This is best estimated from preliminary data (for instance,
        by choosing a ballpark estimate and running a first pass of the analysis and
        observing the resultant histograms for `sigma_x_y` and `sigma_z`).
    :type spot_sigmas: array-like
    :param spot_sigma_xy_bounds: 2-iterable of lower- and upper-bound on acceptable
        standard deviation in microns in x- and y-axes in order for a spot to be considered
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
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :param integrate_sigma_multiple: Multiple of the proposed `spot_sigmas` in all axes
        used to set the dimensions of the ellipsoid that gets integrated over for spot
        quantification.
    :param int expand_distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :param float search_range_um: The maximum distance features in microns can move between
        frames.
    :param int memory: The maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle.
    :param pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type pos_columns: list of DataFrame column names
    :param int averaging: Number of frames to average velocity over.
    :param int min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis.
    :param bool retrack_after_filter: Performs a second tracking after initial filtering
        to avoid tracking getting "distracted" by spurious spots.
    :param retrack_pos_columns: Name of columns in `segmentation_df` containing a position
        coordinate.
    :type retrack_pos_columns: list of DataFrame column names
    :param float retrack_search_range_um: The maximum distance features in microns can move
        between frames during second tracking.
    :param int retrack_memory: The maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle during second tracking.
    :param int retrack_min_track_length: Minimum number of timepoints a spot has to be
        trackable for in order to be considered in the analysis during second tracking.
    :param bool retrack_by_intensity: If `True`, this will attempt to use a preliminary
        spot tracking to use the variation of intensity across traces to help spot
        tracking (i.e. spots with wild jumps in intensity are less likely to be linked
        across frames). See documentation for `normalized_variation_intensity` for
        details.
    :param float retrack_intensity_normalize_quantile: Target value of .84-quantile of
        successive differences in intensity across traces after normalization. This can
        essentially be used to set the "exchange rate" between spatial proximity and
        similarity in intensity (i.e. when tracking, differing in intensity by the
        .84-quantile is penalized as much as being separated by `normalize_quantile_to` from
        a candidate point in the next frame) - the higher the value use, the less tolerant
        tracking is of large variations in intensity.
    :param bool filter_multiple: Decide whether or not to enforce a single-spot
        limit for each nucleus.
    :param bool stitch: If `True`, attempts to stitch together filtered tracks by mean
        position and separation in time.
    :param float stitch_max_distance: Maximum distance between mean position of partial tracks
        that still allows for stitching to occur. If `None`, a default of
        0.5*`retrack_search_range_um` is used.
    :param int stitch_max_frame_distance: Maximum number of frames between tracks with no
        points from either tracks that still allows for stitching to occur.
    :param int stitch_frames_mean: Number of frames to average over when estimating the mean
        position of the start and end of candidate tracks to stitch together.
    :param list series_splits: list of first frame of each series. This is useful
           when stitching together z-coordinates to improve tracking when the z-stack
           has been shifted mid-imaging.
    :param list series_shifts: list of estimated shifts in pixels (sub-pixel
           approximated using centroid of normalized correlation peak) between stacks
           at interface between separate series - this quantifies the shift in the
           z-stack.
    :param float dog_sigma_ratio: Ratio of standard deviations of Difference of Gaussians
        filter used to preprocess the data.
    :param bool evaluate: If `True`, fully evaluates tracked and reordered spot labels
        as a Numpy array after gathering Futures from worker memories.
    :param bool keep_futures: If `True`, keeps a pointer to Futures objects in worker
        memories corresponding to tracked and reordered spot labels, as well as the input
        data.
    :param bool keep_bandpass: If `True`, keeps a copy of the bandpass-filtered image
        in memory. This is kept as `dtype=np.float64`, and on machines lacking in memory
        can cause the Python kernel to crash for larger datasets.
    :param bool keep_spot_labels: If `True`, keeps a copy of the spot mask after thresholding
        but before filtering in memory.

    :ivar default_params: Dictionary of dictionaries, with each subdictionary corresponding
        to the kwargs for one of the functions in the spot analysis pipeline, as described
        in the documentation for `choose_spot_analysis_parameters`.
    :ivar spot_dataframe: pandas DataFrame containing information about putative spots as
        output by :func:`~spot_analysis.detection.detect_and_gather_spots`, along with
        information added by subsequent Gaussian fitting and filtering steps - for details,
        see documentation for `detection.detect_and_gather_spots`,
        `fitting.add_fits_spots_dataframe_parallel`, and
        `track_filtering.track_and_filter_spots`.
    :ivar spot_labels: Boolean mask separating proposed spots in foreground from background.
    :ivar bandpassed_movie: Input spot `data` after bandpass filtering with a Difference
        of Gaussians.
    :ivar reordered_spot_labels: Spot segmentation mask, with labels now corresponding to
        the IDs in the `particle` column of `spot_dataframe`. If available, the IDs are
        transferred over from provided `labels`.
    :ivar reordered_spot_labels_futures: Spot segmentation mask, with labels now
        corresponding to the IDs in the `particle` column of `spot_dataframe`, as a list
        of scattered futures in the Dask Client worker memeories. If available, the IDs are
        transferred over from provided `labels`.

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
        data=None,
        global_metadata=None,
        frame_metadata=None,
        labels=None,
        client=None,
        spot_sigmas=[0.43, 0.21, 0.21],
        spot_sigma_x_y_bounds=(0.052, 0.52),
        spot_sigma_z_bounds=(0.16, 1),
        extract_sigma_multiple=[6, 10, 10],
        num_bootstraps=1000,
        integrate_sigma_multiple=9,
        expand_distance=2,
        search_range_um=4.2,
        memory=2,
        pos_columns=["z", "y", "x"],
        velocity_averaging=None,
        min_track_length=4,
        retrack_after_filter=True,
        retrack_pos_columns=["y", "x"],
        retrack_search_range_um=4.2,
        retrack_memory=6,
        retrack_min_track_length=4,
        retrack_by_intensity=False,
        retrack_intensity_normalize_quantile=0.35,
        filter_multiple=True,
        stitch=True,
        stitch_max_distance=None,
        stitch_max_frame_distance=3,
        stitch_frames_mean=4,
        series_splits=None,
        series_shifts=None,
        dog_sigma_ratio=1.6,
        evaluate=True,
        keep_futures=True,
        keep_bandpass=True,
        keep_spot_labels=True,
    ):
        """
        Constructor method. Instantiates class with no attributes if `data=None`.
        """
        if data is not None:
            self.data = data
            self.global_metadata = global_metadata
            self.frame_metadata = frame_metadata
            self.client = client
            self.spot_sigmas = spot_sigmas
            self.spot_sigma_x_y_bounds = spot_sigma_x_y_bounds
            self.spot_sigma_z_bounds = spot_sigma_z_bounds
            self.extract_sigma_multiple = extract_sigma_multiple
            self.num_bootstraps = num_bootstraps
            self.integrate_sigma_multiple = integrate_sigma_multiple
            self.expand_distance = expand_distance
            self.search_range_um = search_range_um
            self.memory = memory
            self.pos_columns = pos_columns
            self.velocity_averaging = velocity_averaging
            self.min_track_length = min_track_length
            self.retrack_after_filter = retrack_after_filter
            self.retrack_pos_columns = retrack_pos_columns
            self.filter_multiple = filter_multiple
            self.stitch = stitch
            self.stitch_max_distance = stitch_max_distance
            self.stitch_max_frame_distance = stitch_max_frame_distance
            self.stitch_frames_mean = stitch_frames_mean
            self.retrack_search_range_um = retrack_search_range_um
            self.retrack_memory = retrack_memory
            self.retrack_min_track_length = retrack_min_track_length
            self.retrack_by_intensity = retrack_by_intensity
            self.retrack_intensity_normalize_quantile = (
                retrack_intensity_normalize_quantile
            )
            self.series_splits = series_splits
            self.series_shifts = series_shifts
            self.dog_sigma_ratio = dog_sigma_ratio
            self.labels = labels
            self.keep_bandpass = keep_bandpass
            self.keep_spot_labels = keep_spot_labels
            self.image_size = data.shape[1:]

            self.default_params = choose_spot_analysis_parameters(
                channel_global_metadata=self.global_metadata,
                dog_sigma_ratio=self.dog_sigma_ratio,
                spot_sigmas=self.spot_sigmas,
                spot_sigma_x_y_bounds=self.spot_sigma_x_y_bounds,
                spot_sigma_z_bounds=self.spot_sigma_z_bounds,
                extract_sigma_multiple=self.extract_sigma_multiple,
                expand_distance=self.expand_distance,
                search_range_um=self.search_range_um,
                memory=self.memory,
                pos_columns=self.pos_columns,
                velocity_averaging=self.velocity_averaging,
                min_track_length=self.min_track_length,
                filter_multiple=self.filter_multiple,
                retrack_pos_columns=self.retrack_pos_columns,
                retrack_search_range_um=self.retrack_search_range_um,
                retrack_memory=self.retrack_memory,
                retrack_min_track_length=self.retrack_min_track_length,
                retrack_by_intensity=self.retrack_by_intensity,
                retrack_intensity_normalize_quantile=self.retrack_intensity_normalize_quantile,
                num_bootstraps=self.num_bootstraps,
                integrate_sigma_multiple=self.integrate_sigma_multiple,
                keep_bandpass=self.keep_bandpass,
                keep_spot_labels=self.keep_spot_labels,
            )

            self.evaluate = evaluate
            self.keep_futures = keep_futures

    def extract_spot_traces(
        self,
        working_memory_mode="zarr",
        working_memory_folder=None,
        retrack=False,
        rescale=True,
    ):
        """
        Runs through the spot segmentation, tracking, fitting and quantification
        pipeline using the parameters instantiated in the constructor method (or any
        modifications applied to the corresponding class attributes), instantiating a
        class attribute for the output (evaluated and Dask Futures objects) at each
        step if requested.

        :param working_memory_mode: Sets whether the intermediate steps that need to be
            evaluated to construct the nuclear tracking (e.g. construction of a nuclear
            segmentation array) are kept in memory or committed to a `zarr` array. Note
            that using `zarr` as working memory requires the input data to also be a
            `zarr` array, whereby the chunking is inferred from the input data.
        :type working_memory: {"zarr", None}
        :param working_memory_folder: This parameter is required if `working_memory`
            is set to `zarr`, and should be a folder path that points to the location
            where the necessary `zarr` arrays will be stored to disk.
        :type working_memory_folder: {str, `pathlib.Path`, None}
        :param bool retrack: If `True`, only steps subsequent to tracking are re-run.
        :param bool rescale: If `True`, rescales particle positions to correspond
            to real space.
        """
        # If `data` is passed as a zarr array, we wrap it as a list of futures
        # that read each chunk - the parallelization is fully determined by the
        # chunking of the data, so memory management can be done by rechunking
        zarr_in_mode = isinstance(self.data, zarr.core.Array)
        if zarr_in_mode:
            if self.client is not None:
                chunk_boundaries, data = parallel_computing.zarr_to_futures(
                    self.data, self.client
                )
            else:
                data = self.data[:]
                working_memory_mode = None
                warnings.warn("No Dask client available, working in local memory.")
        else:
            data = self.data
            if working_memory_mode == "zarr":
                raise ValueError(
                    "Using `working_memory_mode = 'zarr'` requires the data to be fed in as a `zarr` array."
                )

        if not retrack:
            if working_memory_mode == "zarr":
                self.evaluate = False
                self.keep_spot_labels = False
                self.default_params["detect_and_gather_spots_params"][
                    "return_spot_labels"
                ] = False

                working_memory_path = Path(working_memory_folder)
                results_path = working_memory_path / "spot_analysis_results"
                results_path.mkdir(exist_ok=True)

                spot_labels = zarr.creation.zeros_like(
                    self.data,
                    overwrite=True,
                    store=results_path / "spot_labels.zarr",
                    dtype=np.uint32,
                )

                reordered_spot_labels = zarr.creation.zeros_like(
                    self.data,
                    overwrite=True,
                    store=results_path / "reordered_spot_labels.zarr",
                    dtype=np.uint32,
                )

                warnings.warn(
                    "`working_memory_mode` set to 'zarr', will not pull `spot_labels` into memory."
                )

            elif working_memory_mode is not None:
                raise ValueError("`working_memory_mode` option not recognized.")

            # We set `return_spot_dataframe` to False without explicitly considering
            # whether the operation is being sent to a cluster since
            # `detection.detect_and_gather_spots" internally handles that and fully
            # evaluates the dataframe if no client is supplied.
            (
                self.spot_dataframe,
                self.spot_dataframe_futures,
                self.spot_labels,
                self.spot_labels_futures,
                self.bandpassed_movie,
                self.bandpassed_movie_futures,
                self.spot_movie_futures,
            ) = detection.detect_and_gather_spots(
                data,
                frame_metadata=self.frame_metadata,
                keep_futures_spot_labels=True,
                keep_futures_spots_movie=self.keep_futures,
                keep_futures_spot_dataframe=True,
                keep_futures_bandpass=self.keep_futures,
                return_spot_dataframe=False,
                client=self.client,
                **(self.default_params["detect_and_gather_spots_params"]),
            )

            # Dump to zarr here to avoid memory bottleneck since tracking is a blocking
            # operation taking place in the local process. We handled checking for
            # parallelization earlier, no need to do that here.
            if working_memory_mode == "zarr":
                parallel_computing.futures_to_zarr(
                    self.spot_labels_futures, chunk_boundaries, spot_labels, self.client
                )
                self.spot_labels = spot_labels

                if not self.keep_futures:
                    del self.spot_labels_futures
                    self.spot_labels_futures = None

            if self.client is not None:
                add_fits_func = partial(
                    fitting.add_fits_spots_dataframe,
                    image_size=self.image_size,
                    **(self.default_params["add_fits_spots_dataframe_parallel_params"]),
                )

                self.spot_dataframe_futures = self.client.map(
                    add_fits_func, self.spot_dataframe_futures
                )

                add_neighborhood_intensity_func = partial(
                    fitting.add_neighborhood_intensity_spot_dataframe,
                    **(
                        self.default_params[
                            "add_neighborhood_intensity_spot_dataframe_parallel_params"
                        ]
                    ),
                )

                self.spot_dataframe_futures = self.client.map(
                    add_neighborhood_intensity_func, self.spot_dataframe_futures
                )

                # If labels are provided, we transfer nuclear labels before proceeding
                if (self.labels is not None) and (working_memory_mode == "zarr"):
                    zarr_labels_mode = isinstance(self.labels, zarr.core.Array)

                    if not zarr_labels_mode:
                        labels = zarr.creation.array(
                            self.labels,
                            overwrite=True,
                            store=results_path / "nuclear_labels.zarr",
                            dtype=np.uint32,
                        )
                    else:
                        labels = self.labels

                    transfer_labels_func = partial(
                        _transfer_labels,
                        labels=labels,
                        params=self.default_params,
                        pos_columns=self.pos_columns,
                    )

                    self.spot_dataframe_futures = self.client.map(
                        transfer_labels_func, self.spot_dataframe_futures
                    )

                spot_dataframe = pd.concat(
                    self.client.gather(self.spot_dataframe_futures),
                    ignore_index=True,
                )

                if not self.keep_futures:
                    del self.spot_dataframe_futures
                    self.spot_dataframe_futures = None

                # If the dataframes characterizing spots in each chunk (`Future`) of the
                # movie are constructed independently in each worker, we use the field
                # `original_frame` to keep track of the frame number in the full movie
                # and `frame` to keep track of the frame number within each chunk. This
                # is swapped out and `original_frame` is dropped when the chunked
                # dataframes are gathered back to the local process.
                spot_dataframe.drop(labels=["frame"], axis=1, inplace=True)
                spot_dataframe.rename({"original_frame": "frame"}, axis=1, inplace=True)
                self.spot_dataframe = spot_dataframe

            else:
                self.spot_dataframe = fitting.add_fits_spots_dataframe(
                    self.spot_dataframe,
                    **(self.default_params["add_fits_spots_dataframe_parallel_params"]),
                )

                self.spot_dataframe = fitting.add_neighborhood_intensity_spot_dataframe(
                    self.spot_dataframe,
                    **(
                        self.default_params[
                            "add_neighborhood_intensity_spot_dataframe_parallel_params"
                        ]
                    ),
                )

        if rescale:
            # Back up pixel-space positions
            pixels_column_names = ["".join([pos, "_pixel"]) for pos in self.pos_columns]
            for i, _ in enumerate(pixels_column_names):
                self.spot_dataframe[pixels_column_names[i]] = self.spot_dataframe[
                    self.pos_columns[i]
                ].copy()

            # Account for z-stack shift between series
            if (self.series_splits is not None) and ("z" in self.pos_columns):
                for i, shift_frame in enumerate(self.series_splits):
                    self.spot_dataframe.loc[
                        self.spot_dataframe["frame"] > shift_frame, "z"
                    ] = (
                        self.spot_dataframe.loc[
                            self.spot_dataframe["frame"] > shift_frame, "z"
                        ]
                        - self.series_shifts[i]
                    )

            # Rescale pixel-space position columns to match real space
            if self.global_metadata is not None:
                mpp_pos_fields = [
                    "".join(["PixelsPhysicalSize", pos.capitalize()])
                    for pos in self.pos_columns
                ]
                mpp_pos = [self.global_metadata[field] for field in mpp_pos_fields]

                for i, _ in enumerate(self.pos_columns):
                    self.spot_dataframe[self.pos_columns[i]] = (
                        self.spot_dataframe[self.pos_columns[i]] * mpp_pos[i]
                    )

        track_filtering.track_and_filter_spots(
            self.spot_dataframe,
            nuclear_labels=self.labels,
            nuclear_pos_columns=pixels_column_names,
            client=self.client,
            **(self.default_params["track_and_filter_spots_params"]),
        )

        # A second tracking allows tracking to occur without distraction from spurious
        # particles that end up getting culled anyway - this allows for use of a larger
        # search radius and memory parameter if desired. This is only used if nuclear
        # labels are not provided, otherwise the spot tracking is not actually used to
        # assign particle IDs.
        if self.retrack_after_filter and (self.labels is None):
            filtered_dataframe = self.spot_dataframe[
                self.spot_dataframe["particle"] != 0
            ].copy()

            track_filtering.track_and_filter_spots(
                filtered_dataframe,
                nuclear_labels=self.labels,
                nuclear_pos_columns=pixels_column_names,
                client=self.client,
                **(self.default_params["retrack_spots_params"]),
            )

            self.spot_dataframe.loc[
                filtered_dataframe.index, "particle"
            ] = filtered_dataframe["particle"]

        if self.stitch:
            print("Stitching tracks.")
            if self.stitch_max_distance is None:
                self.stitch_max_distance = 0.5 * self.search_range_um

            stitch_tracks.stitch_tracks(
                self.spot_dataframe,
                self.retrack_pos_columns,
                self.stitch_max_distance,
                self.stitch_max_frame_distance,
                self.stitch_frames_mean,
                quantification=self.default_params["track_and_filter_spots_params"][
                    "quantification"
                ],
                inplace=True,
            )

        if rescale:
            # Restore dataframe to pixel-space
            for i, _ in enumerate(self.pos_columns):
                self.spot_dataframe[self.pos_columns[i]] = self.spot_dataframe[
                    pixels_column_names[i]
                ]
                self.spot_dataframe = self.spot_dataframe.drop(
                    pixels_column_names[i], axis=1
                )

        # Pull from zarr here

        if self.client is not None:
            if working_memory_mode == "zarr":
                first_last_frames = np.array(chunk_boundaries) + 1
                first_last_frames[:, 1] -= 1
                first_last_frames = first_last_frames.tolist()

                if self.spot_labels_futures is None:
                    _, self.spot_labels_futures = parallel_computing.zarr_to_futures(
                        self.spot_labels, self.client
                    )
            else:
                first_last_frames = None

            (
                self.reordered_spot_labels,
                self.reordered_spot_labels_futures,
                _,
            ) = track_features.reorder_labels_parallel(
                self.spot_labels_futures,
                self.spot_dataframe,
                client=self.client,
                first_last_frames=first_last_frames,
                futures_in=False,
                futures_out=True,
                evaluate=self.evaluate,
            )

            if working_memory_mode == "zarr":
                parallel_computing.futures_to_zarr(
                    self.reordered_spot_labels_futures,
                    chunk_boundaries,
                    reordered_spot_labels,
                    self.client,
                )
                self.reordered_spot_labels = reordered_spot_labels

            if not self.keep_futures:
                del self.spot_labels_futures
                self.spot_labels_futures = None

                del self.reordered_spot_labels_futures
                self.reordered_spot_labels_futures = None

        else:
            self.reordered_spot_labels = track_features.reorder_labels(
                self.spot_labels, self.spot_dataframe
            )
            self.reordered_spot_labels_futures = None

    def save_results(self, *, name_folder, save_array_as="zarr", save_all=False):
        """
        Saves results of spot segmentation and tracking to disk as HDF5, with the
        labelled segmentation mask saved as zarr or TIFF as specified.

        :param str file_name: Name to save reordered (post-tracking) labelled spot
            mask under.
        :param str name_folder: Name of folder to create `spot_analysis_results`
            subdirectory in, in which analysis results are stored.
        :param save_array_as: Format to save reordered spot labels in:
        :type save_array_as: {"zarr", "tiff"}
        :param bool save_all: If true, saves a tiff file for each intermediate step
            of the spot analysis pipeline
        """
        # Make `spot_analysis_results` directory if it doesn't exist
        name_path = Path(name_folder)
        results_path = name_path / "spot_analysis_results"
        results_path.mkdir(exist_ok=True, parents=True)

        # Save movies, saving intermediate steps if requested
        save_movies = {"reordered_spot_labels": self.reordered_spot_labels}

        if save_all:
            save_movies["spot_labels"] = self.spot_labels
            save_movies["bandpassed_movie"] = self.bandpassed_movie

        file_not_found = "".join(
            [
                " could not be found, check to make",
                " sure the `keep_bandpass` and `keep_spot_labels`",
                " kwargs were used when running the",
                " `extract_spot_traces` method so that the",
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

        elif save_array_as is not None:
            raise ValueError("`save_array_as` option not recognized.")

        # Save spot dataframe
        spot_dataframe_path = results_path / "spot_dataframe.pkl"
        self.spot_dataframe.to_pickle(spot_dataframe_path)

        # Keep a copy of the parameters used
        spot_analysis_param_path = results_path / "spot_analysis_parameters.pkl"
        with open(spot_analysis_param_path, "wb") as f:
            pickle.dump(self.default_params, f)

    def read_results(self, *, name_folder, import_all=False):
        """
        Imports results from a saved run of `extract_spot_traces` into the corresponding
        class attributes.

        :param str name_folder: Name of folder where `spot_analysis_results`
            subdirectory resides.
        :param bool import_all: If `True`, will attempt to also import results of
            intermediate steps of spot analysis if they have been saved to disk,
            showing a warning if a corresponding file cannot be found in the specified
            directory.
        """
        name_path = Path(name_folder)
        results_path = name_path / "spot_analysis_results"

        # Import arrays
        import_arrays = ["reordered_spot_labels"]
        if import_all:
            import_arrays.extend(
                [
                    "spot_labels",
                    "bandpassed_movie",
                ]
            )

        for array in import_arrays:
            # Try importing as zarr first - if there is no zarr array, this will
            # return `None`
            zarr_path = results_path / "".join([array, ".zarr"])
            setattr(self, array, zarr.open(zarr_path))

            if getattr(self, array) is None:
                tiff_path = results_path / "".join([array, ".tiff"])
                try:
                    setattr(self, array, imread(tiff_path, plugin="tifffile"))
                except FileNotFoundError:
                    warnings.warn("".join([array, " not found, keeping as `None`."]))

        # Import dataframe
        dataframe_path = results_path / "spot_dataframe.pkl"
        try:
            self.spot_dataframe = pd.read_pickle(dataframe_path, compression=None)
        except FileNotFoundError:
            self.spot_dataframe = None
            warnings.warn("`spot_dataframe` not found, keeping as `None`.")

        # Import saved parameters
        with open(results_path / "spot_analysis_parameters.pkl", "rb") as f:
            self.default_params = pickle.load(f)