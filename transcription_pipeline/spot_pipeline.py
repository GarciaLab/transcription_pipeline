from .tracking import track_features
from .spot_analysis import detection, fitting, track_filtering
from scipy.optimize import fsolve
import warnings
import numpy as np
import pandas as pd
import zarr
from skimage.io import imsave, imread
from pathlib import Path
import pickle
import glob


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
    keep_spot_labels,
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
            self.dog_sigma_ratio = dog_sigma_ratio
            self.labels = labels
            self.keep_bandpass = keep_bandpass
            self.keep_spot_labels = keep_spot_labels

            self.default_params = choose_spot_analysis_parameters(
                self.global_metadata,
                self.spot_sigmas,
                self.spot_sigma_x_y_bounds,
                self.spot_sigma_z_bounds,
                self.extract_sigma_multiple,
                self.dog_sigma_ratio,
                self.keep_bandpass,
                self.keep_spot_labels,
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
            self.spot_labels,
            self.spot_labels_futures,
            self.bandpassed_movie,
            self.bandpassed_movie_futures,
        ) = detection.detect_and_gather_spots(
            self.data,
            frame_metadata=self.frame_metadata,
            keep_futures_spot_labels=True,
            keep_futures_bandpass=self.keep_futures,
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
            nuclear_labels=self.labels,
            client=self.client,
            **(self.default_params["track_and_filter_spots_params"]),
        )

        (
            self.reordered_spot_labels,
            self.reordered_spot_labels_futures,
            _,
        ) = track_features.reorder_labels_parallel(
            self.spot_labels_futures,
            self.spot_dataframe,
            client=self.client,
            futures_in=False,
            futures_out=self.keep_futures,
            evaluate=self.evaluate,
        )

        if not self.keep_futures:
            del self.spot_labels_futures
            self.spot_labels_futures = None

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
        results_path.mkdir(exist_ok=True)

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
            setattr(self, array, zarr.load(zarr_path))

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