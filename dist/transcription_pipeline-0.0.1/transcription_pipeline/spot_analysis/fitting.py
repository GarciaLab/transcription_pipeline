from ..utils import neighborhood_manipulation
from scipy.optimize import least_squares
from scipy.linalg import svd
import numpy as np
from functools import partial
import dask.dataframe as dd
import deprecation


def gaussian3d_sym_xy(coordinates, *, centroid, sigma_x_y, sigma_z, amplitude, offset):
    """
    Evaluates a 3d Gaussian function isotropic in the xy-plane and with specified
    centroid at specified coordinates.

    :param coordinates: Coordinates at which to evaluate the Gaussian.
    :type coordinates: np.ndarray
    :param centroid: Coordinates of centroid of Gaussian function to evaluate.
    :type centroid: np.ndarray
    :param float sigma_x_y: Standard deviation of Gaussian function in the x- and y-
        coordinate direction.
    :param float sigma_z: Standard deviation of Gaussian function in the z-coordinate.
    :param float amplitude: Amplitude of Gaussian function.
    :param float offset: Offset (limit away from centroid) of Gaussian.
    :return: Evaluated Gaussian.
    :rtype: float
    """
    z, y, x = coordinates
    z_0, y_0, x_0 = centroid
    gaussian_eval = offset + amplitude * np.exp(
        -(((x - x_0) ** 2) / (2 * sigma_x_y**2))
        - (((y - y_0) ** 2) / (2 * sigma_x_y**2))
        - (((z - z_0) ** 2) / (2 * sigma_z**2))
    )
    return gaussian_eval


def generate_gaussian_3d_sym_xy(
    box_shape, centroid, sigma_x_y, sigma_z, amplitude, offset, coordinate_array=None
):
    """
    Evaluates a 3d Gaussian function isotropic in the xy-plane and with specified
    centroid at specified coordinates.

    :param box_shape: Shape of bounding box containing the generated Gaussian.
    :type box_shape: {np.ndarray, tuple[int], list[int]}
    :param centroid: Coordinates of centroid of Gaussian function to evaluate.
    :type centroid: {np.ndarray, tuple[float], list[float]}
    :param float sigma_x_y: Standard deviation of Gaussian function in the x- and y-
        coordinate direction.
    :param float sigma_z: Standard deviation of Gaussian function in the z-coordinate.
    :param float amplitude: Amplitude of Gaussian function.
    :param float offset: Offset (limit away from centroid) of Gaussian.
    :param coordinate_array: If `None`, it is assumed that the pixel data is ordered by
        the usual `tzyx` axes. Otherwise, an array of coordinates of the same shape
        as `data` can be passed to specify the coordinates of each pixel.
    :type coordinate_array: np.ndarray
    :return: Generated Gaussian inside bounding box.
    :rtype: np.ndarray
    """
    if coordinate_array is None:
        coordinate_array = np.indices(box_shape) + 0.5

    centroid = np.asarray(centroid)

    gaussian_box = gaussian3d_sym_xy(
        coordinate_array,
        centroid=centroid,
        sigma_x_y=sigma_x_y,
        sigma_z=sigma_z,
        amplitude=amplitude,
        offset=offset,
    )

    return gaussian_box


def _perimeter_mask(data):
    """
    Creates a boolean mask that selects the values of the outside shell of a numpy
    array.
    """
    shape = np.asarray(data.shape)
    border_mask = np.ones(shape, dtype=bool)
    inside_mask = np.zeros(shape - 2, dtype=bool)
    neighborhood_manipulation.inject_neighborhood(border_mask, inside_mask, 1)
    return border_mask


def _extract_perimeter_mean(data):
    """
    Calculates the mean of values on the outside surface of input numpy array `data`.
    """
    return data[_perimeter_mask(data)].mean()


def fit_gaussian_3d_sym_xy(
    data,
    *,
    centroid_guess,
    sigma_x_y_guess,
    sigma_z_guess,
    coordinate_array=None,
    amplitude_guess=None,
    offset_guess=None,
    method="trf",
    **kwargs,
):
    """
    Fits an xy-symmetric 3D Gaussian to `data`, returning fit parameters and other
    relevant fit information.

    :param data: 3-dimensional array (with the usual spatial axis ordering 'zyx')
        containing a single putative spot to fit with a 3D Gaussian.
    :type data: np.ndarray
    :param centroid_guess: Initial guess for Gaussian centroid to feed to least
        squares minimization.
    :type centroid_guess: np.ndarray
    :param float sigma_x_y_guess: Initial guess for standard deviation of Gaussian
        fit in x- and y-coordinates, assumed to be symmetric.
    :param float sigma_z_guess: Initial guess for standard deviation of Gaussian
        fit in z-coordinate.
    :param coordinate_array: If `None`, it is assumed that the pixel data is ordered by
        the usual `tzyx` axes. Otherwise, an array of coordinates of the same shape
        as `data` can be passed to specify the coordinates of each pixel.
    :type coordinate_array: np.ndarray
    :param float amplitude_guess: Initial guess for amplitude of Gaussian fit. If
        None (default), an initial guess will be computed from the data by taking the
        maximum.
    :param float offset_guess: Initial guess for offset of Gaussian fit - this is
        useful for estimating the background for the detected spot. If None (default),
        an initial guess will be computed from the data by taking the mean value of
        the voxels on the surface of the `data` array (incidentally, this seems to
        already be a reasonable estimate of background).
    :param str method: Method to use for least-squares optimization (see
        `scipy.optimize.least_squares`).
    :return: If optimization is successful, returns a tuple of fit parameters
        `(centroid, sigma_x_y, sigma_z, amplitude, offset, cost)`. The notation
        is consistent with that used in `gaussian3d_sym_xy`.

        * `centroid`: Centroid of fitted 3D Gaussian (Numpy array).
        * `sigma_x_y`: Standard deviation of fitted 3D Gaussian in x- and y-coordinate.
        * `sigma_z`: Standard deviation of fitted 3D Gaussian in z-coordinate.
        * `amplitude`: Amplitude of fitted 3D Gaussian.
        * `offset`: Offset of fitted 3D Gaussian.
        * `cost`: Value of cost function at the solution.

    :rtype: tuple

    .. note:: This function can also pass through any kwargs accepted by
        `scipy.optimization.least_squares`.
    """
    error_out = (np.full(3, np.nan),) + (np.nan,) * 6

    if data is None:
        out = error_out

    else:
        if offset_guess is None:
            offset_guess = _extract_perimeter_mean(data)

        if amplitude_guess is None:
            amplitude_guess = data.max() - offset_guess

        param_initial = np.array(
            [
                *centroid_guess,
                sigma_x_y_guess,
                sigma_z_guess,
                amplitude_guess,
                offset_guess,
            ]
        )

        def residuals_func(x):
            """
            Helper function to compute the residuals of a 3D Gaussian fit.
            """
            residuals = (
                generate_gaussian_3d_sym_xy(
                    data.shape,
                    [x[0], x[1], x[2]],
                    *x[3:],
                    coordinate_array=coordinate_array,
                )
                - data
            ).ravel()
            return residuals

        result = least_squares(
            residuals_func, param_initial, bounds=(0, np.inf), method=method, **kwargs
        )

        if not result.success:
            out = error_out
        else:
            params = result.x
            centroid = params[:3]
            sigma_x_y = params[3]
            sigma_z = params[4]
            amplitude = params[5]
            offset = params[6]
            cost = result.cost

            # Taken from https://stackoverflow.com/a/67023688
            # noinspection PyTupleAssignmentBalance
            U, s, Vh = svd(result.jac, full_matrices=False)
            tol = np.finfo(float).eps * s[0] * max(result.jac.shape)
            w = s > tol
            cov = (Vh[w].T / s[w] ** 2) @ Vh[w]  # robust covariance matrix
            # Rescale covariance assuming reduced chi-sq is unity (assumes equal errors)
            # This should be removed if we implement pixel-by-pixel error estimates
            chi2dof = np.sum(result.fun**2) / (result.fun.size - result.x.size)
            cov *= chi2dof

            out = (centroid, sigma_x_y, sigma_z, amplitude, offset, cost, cov)

    return out


def _fit_spot_dataframe_row(
    row,
    *,
    sigma_x_y_guess,
    sigma_z_guess,
    amplitude_guess,
    offset_guess,
    method,
    image_size,
    **kwargs,
):
    """
    Fits a 3D Gaussian to the raw data in a row of the DataFrame of detected spots
    output by :func:`~spot_analysis.detection.detect_and_gather_spots` using
    :func:`~fit_gaussian_3d_sym_xy`.
    """
    spot_data = row["raw_spot"]
    spatial_coordinates_start = row["coordinates_start"][1:]
    centroid_guess = row[["z", "y", "x"]].values - spatial_coordinates_start

    fit_results = fit_gaussian_3d_sym_xy(
        spot_data,
        centroid_guess=centroid_guess,
        sigma_x_y_guess=sigma_x_y_guess,
        sigma_z_guess=sigma_z_guess,
        amplitude_guess=amplitude_guess,
        offset_guess=offset_guess,
        method=method,
        **kwargs,
    )

    fit_relative_centroid = fit_results[0]

    # Restore initial centroid values if fit failed
    if np.isnan(fit_relative_centroid).any():
        fit_relative_centroid = centroid_guess

    refined_centroid = spatial_coordinates_start + fit_relative_centroid

    # Restore initial centroid values if refining would put the centroid outside the
    # FOV
    if image_size is not None:
        if np.any(refined_centroid > (np.asarray(image_size) - 1)):
            refined_centroid = centroid_guess + spatial_coordinates_start

    out = (*refined_centroid, *fit_results[1:])

    return out


def intensity_from_fit_row(spot_dataframe_row):
    """
    Uses the analytical expression for the integral of a 3D xy-symmetric Gaussian
    over space to estimate the integrated spot brightness from the amplitude and
    sigma fit parameters.

    :param spot_dataframe_row: Row of DataFrame containing information about putative
        spots as output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe_row: row of pandas DataFrame

    .. note::

        .. math::

            \\int_{\\mathbb{R}^3} A e^{- \\frac{x^2 + y^2}{2 \\sigma_{xy}^2}
            - \\frac{z^2}{2 \\sigma_z^2}} \\ dx \\ dy \\ dz =
            2 \\sqrt{2} A \\pi^{3/2} \\sigma_{xy}^2 \\sigma_z

    """
    amplitude = spot_dataframe_row["amplitude"]
    sigma_x_y = spot_dataframe_row["sigma_x_y"]
    sigma_z = spot_dataframe_row["sigma_z"]
    integrated_amplitude = (
        2 * np.sqrt(2) * amplitude * (np.pi ** (3 / 2)) * (sigma_x_y**2) * sigma_z
    )
    return integrated_amplitude


def intensity_error_from_fit_row(spot_dataframe_row):
    """
    Uses the analytical expression for integrated intensity from `intensity_from_fit_row`
    to estimate the standard error of the intensity from the covariance matrix of the
    fit parameters.

    :param spot_dataframe_row: Riw of DataFrame containing information about putative spots
        a output by :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe_row: Row of pandas DataFrame

    .. note::

        We make use of the first-order expansion of errors for :math:`I = \prod_i X_i` given
        by:

        .. math::
            \\bigg( \\frac{\\sigma_I}{I} \\bigg)^2 = \\sum_i \\bigg( \\frac{\\sigma_{X_i}}{X_i})^2 \\bigg) \\
            + 2 \\sum_i \\sum_{j > i} \\frac{\\sigma_{X_i X_j}}{X_i X_j}

    """
    amplitude = spot_dataframe_row["amplitude"]
    sigma_x_y = spot_dataframe_row["sigma_x_y"]
    sigma_z = spot_dataframe_row["sigma_z"]
    cov = spot_dataframe_row["covariance_matrix"]
    frac_err = np.sqrt(
        (cov[5, 5] / (amplitude**2))
        + 4 * (cov[3, 3] / (sigma_x_y**2))
        + (cov[4, 4] / (sigma_z**2))
        + 2
        * (
            2 * (cov[5, 3] / (amplitude * sigma_x_y))
            + 2 * (cov[3, 4] / (sigma_z * sigma_x_y))
            + (cov[5, 4] / (amplitude * sigma_z))
        )
    )
    intensity = spot_dataframe_row["intensity_from_fit"]
    err_intensity = intensity * frac_err
    return err_intensity


def add_fits_spots_dataframe(
    spot_df,
    *,
    sigma_x_y_guess,
    sigma_z_guess,
    amplitude_guess=None,
    offset_guess=None,
    image_size=None,
    method="trf",
    inplace=True,
    **kwargs,
):
    """
    Fits a 3D Gaussian to the raw data in each row of the DataFrame of detected spots
    output by :func:`~spot_analysis.detection.detect_and_gather_spots` using
    :func:`~fit_gaussian_3d_sym_xy`. The fit centroid is then used to refine the
    spatial coordinates of the spot, and columns are added as follows:

    * "sigma_x_y": Standard deviation of Gaussian fit in x- and y-coordinates.
    * "sigma_z": Standard deviation of Gaussian fit in z-coordinate.
    * "amplitude": Amplitude of Gaussian fit (this is typically the signal of interest).
    * "offset": Offset of Gaussian fit (useful for background subtraction).
    * "cost": Value of the cost function at termination of the fit, see documentation
      for `scipy.optimize.least_squares`.
    * "norm_cost": Normalized cost, defined as the L2-norm of the residuals divided
      by the product of the amplitude and the number of voxels. This gives a dimensionless
      measure of cost of the fit that can be more easily used for downstream filtering.
    * "intensity_from_fit": Estimated spot intensity by using analytical expression for
      integral of 3D Gaussian over space and fit parameters.

    :param spot_df: DataFrame containing information about putative spots as output by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_df: pandas DataFrame
    :param float sigma_x_y_guess: Initial guess for standard deviation of Gaussian
        fit in x- and y-coordinates, assumed to be symmetric.
    :param float sigma_z_guess: Initial guess for standard deviation of Gaussian
        fit in z-coordinate.
    :param float amplitude_guess: Initial guess for amplitude of Gaussian fit. If
        None (default), an initial guess will be computed from the data by taking the
        maximum.
    :param float offset_guess: Initial guess for offset of Gaussian fit - this is
        useful for estimating the background for the detected spot. If None (default),
        an initial guess will be computed from the data by taking the mean value of
        the voxels on the surface of the `data` array (incidentally, this seems to
        already be a reasonable estimate of background).
    :param image_size: Shape of the array corresponding to a frame. This is used to
        check whether the proposed spot centroids are within the image bounds.
    :type image_size: np.ndarray
    :param str method: Method to use for least-squares optimization (see
        `scipy.optimize.least_squares`).
    :param bool inplace: If True, the input `spot_df` is modified in-place to add the
        required columns and returns `None`. Otherwise, a modified copy is returned.
    :return: If `inplace=False`, returns copy of input `spot_df` with added columns
        for fit characteristics of the 3D Gaussian fit. Otherwise returns `None`.
    :rtype: {pandas DataFrame, None}

    .. note::
        This function can also pass through any kwargs taken by
        `scipy.optimize.least_squares`.
    """
    if inplace:
        spot_dataframe = spot_df
        out = None
    else:
        spot_dataframe = spot_df.copy()
        out = spot_dataframe

    fit_spot_row_func = partial(
        _fit_spot_dataframe_row,
        sigma_x_y_guess=sigma_x_y_guess,
        sigma_z_guess=sigma_z_guess,
        amplitude_guess=amplitude_guess,
        offset_guess=offset_guess,
        method=method,
        image_size=image_size,
        **kwargs,
    )

    fit_results = spot_dataframe.apply(fit_spot_row_func, axis=1)

    spot_dataframe["covariance_matrix"] = None
    spot_dataframe["covariance_matrix"].astype(object)

    fit_fields = [
        "z",
        "y",
        "x",
        "sigma_x_y",
        "sigma_z",
        "amplitude",
        "offset",
        "cost",
        "covariance_matrix",
    ]
    for i, field in enumerate(fit_fields):
        spot_dataframe[field] = fit_results.apply(lambda x: x[i])

    # Add normalized cost as a dimensionless metric for easier filtering based o
    # goodness-of-fit. We multiply the cost function by 2 to cancel out the factor of
    # 1/2 that `scipy.optimize.least_squares` adds to the cost (it does this to get a nicer
    # algebraic form for the gradient, but we readjust it to keep consistent with the
    # L2 norm definition).
    attempted_fit_mask = ~spot_dataframe["cost"].apply(np.isnan)

    # Initialize columns and check if we have any usable spots
    spot_dataframe["norm_cost"] = np.nan
    spot_dataframe["intensity_from_fit"] = np.nan
    spot_dataframe["intensity_std_error_from_fit"] = np.nan

    if attempted_fit_mask.sum() > 0:
        dataset_size = spot_dataframe.loc[attempted_fit_mask, "raw_spot"].apply(
            lambda x: x.size
        )
        amplitude = spot_dataframe.loc[attempted_fit_mask, "amplitude"]

        spot_dataframe.loc[attempted_fit_mask, "norm_cost"] = (
            (2 * spot_dataframe.loc[attempted_fit_mask, "cost"])
            .apply(np.sqrt)
            .divide(dataset_size * amplitude)
        )

        # Add intensity estimates from fit parameters
        spot_dataframe.loc[attempted_fit_mask, "intensity_from_fit"] = spot_dataframe[
            attempted_fit_mask
        ].apply(intensity_from_fit_row, axis=1)

        # Add intensity error estimates from fit parameters
        spot_dataframe["intensity_std_error_from_fit"] = spot_dataframe[
            attempted_fit_mask
        ].apply(intensity_error_from_fit_row, axis=1)

    return out


def add_fits_spots_dataframe_parallel(
    spot_dataframe,
    *,
    sigma_x_y_guess,
    sigma_z_guess,
    client,
    amplitude_guess=None,
    offset_guess=None,
    method="trf",
    evaluate=True,
    inplace=True,
    **kwargs,
):
    """
    Parallelizes :func:`~add_fits_spots_dataframe` across a Dask Cluster.

    :param spot_dataframe: DataFrame containing information about putative spots as output by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe: pandas DataFrame
    :param float sigma_x_y_guess: Initial guess for standard deviation of Gaussian
        fit in x- and y-coordinates, assumed to be symmetric.
    :param float sigma_z_guess: Initial guess for standard deviation of Gaussian
        fit in z-coordinate.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param float amplitude_guess: Initial guess for amplitude of Gaussian fit. If
        None (default), an initial guess will be computed from the data by taking the
        maximum.
    :param float offset_guess: Initial guess for offset of Gaussian fit - this is
        useful for estimating the background for the detected spot. If None (default),
        an initial guess will be computed from the data by taking the mean value of
        the voxels on the surface of the `data` array (incidentally, this seems to
        already be a reasonable estimate of background).
    :param str method: Method to use for least-squares optimization (see
        `scipy.optimize.least_squares`).
    :param bool evaluate: If True, returns a fully-evaluated modified copy of the input
        `spot_dataframe` with the required columns added. Otherwise, returns a pointer
        to a Dask task that can be evaluated and returned on demand using the `compute`
        method. Note that `inplace=True` forces evaluation regardless of this parameter.
    :param bool inplace: If True, the input `spot_df` is modified in-place to add the
        required columns and returns `None`. Otherwise, a modified copy is returned.
    :return: If `inplace=False`, returns copy of input `spot_df` with added columns
        for fit characteristics of the 3D Gaussian fit. Otherwise returns `None`.
    :rtype: {pandas DataFrame, None}

    .. note::
        This function can also pass through any kwargs taken by
        `scipy.optimize.least_squares`.
    """
    # Preallocate columns to facilitate sharing metadata with Dask
    spot_dataframe[
        [
            "sigma_x_y",
            "sigma_z",
            "amplitude",
            "offset",
            "cost",
            "norm_cost",
            "intensity_from_fit",
        ]
    ] = np.nan
    num_processes = len(client.scheduler_info()["workers"])
    spot_dataframe_dask = dd.from_pandas(spot_dataframe, npartitions=num_processes)
    add_fits_spot_func = partial(
        add_fits_spots_dataframe,
        sigma_x_y_guess=sigma_x_y_guess,
        sigma_z_guess=sigma_z_guess,
        amplitude_guess=amplitude_guess,
        offset_guess=offset_guess,
        method=method,
        inplace=False,
        **kwargs,
    )

    # Map across partitions
    spot_dataframe_dask = spot_dataframe_dask.map_partitions(
        add_fits_spot_func, meta=spot_dataframe_dask
    )

    if evaluate:
        spot_dataframe_dask = client.compute(spot_dataframe_dask)

    if inplace:
        spot_dataframe[:] = spot_dataframe_dask
        out = None
    else:
        out = spot_dataframe_dask

    return out


def extract_spot_shell(
    raw_spot, *, centroid, mppZ, mppYX, ball_diameter_um, shell_width_um, aspect_ratio
):
    """
    Extracts pixel values within an ellipsoid neighborhood around a proposed spot, and
    pixel values in a shell around that neighborhood for spot and background
    quantification respectively.

    :param raw_spot: Cuboidal neighborhood around a spot as extracted by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type raw_spot: np.ndarray
    :param centroid: Centroid of spot, usually obtained by Gaussian fitting.
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :return: (spot_values, background_values) where `spot` is an array of the pixel values
        inside the ellipsoid neighborhood around the spot, and `background` is an
        array of the pixel values in the shell around the neighborhood.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Compute distance in pixel space
    indices = np.indices(raw_spot.shape, dtype=float) + 0.5
    distance_pixels = (indices.T - np.asarray(centroid)).astype(float)

    # Use scope resolution to transform to real space
    distance_pixels[..., 0] *= mppZ * aspect_ratio
    distance_pixels[..., 1:] *= mppYX
    distance_grid = np.sqrt(np.sum(distance_pixels**2, axis=3)).T

    # Extract mask for ball centered around centroid and shell around ball
    ball_mask = distance_grid < (ball_diameter_um / 2)
    shell_mask = (distance_grid >= (ball_diameter_um / 2)) & (
        distance_grid < (ball_diameter_um / 2) + shell_width_um
    )
    spot = raw_spot[ball_mask]
    background = raw_spot[shell_mask]

    return spot, background


@deprecation.deprecated(
    details="Use `bootstrap_intensity` instead of `simple_bootstrap_intensity`, see documentation for details.."
)
def simple_bootstrap_intensity(
    raw_spot,
    *,
    centroid,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
    num_bootstraps=1000,
):
    """
    Extracts pixel values within an ellipsoid neighborhood around a proposed spot, and
    pixel values in a shell around that neighborhood for spot and background
    quantification respectively, then estimates the intensity as the sum of spot
    pixel values in the ellipsoid mask, background-subtracted by estimating the background
    intensity using the shell around the ellipsoid spot mask. This procedure is bootstrapped
    and used to estimate the error on the intensity.

    :param raw_spot: Cuboidal neighborhood around a spot as extracted by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type raw_spot: np.ndarray
    :param centroid: Centroid of spot, usually obtained by Gaussian fitting.
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the  xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :return: (intensity, intensity_err) where `intensity` is the sum of pixel values
        inside the ellipsoid spot mask, background-subtracted by estimating the average
        background intensity per pixel from the shell around the ellipsoid mask, and
        averaged over `num_bootstraps` bootstrap samples. `intensity_err` is the standard
        deviation of the same.
    :rtype: tuple

    .. note:: If the imaging settings are fast relative to the diffusion time of
        transcriptional loci, a neighborhood of ~3 sigmas is sufficient to obtain
        good quantification of the spot. Otherwise (as they are on our system, with
        ~0.6s between z-slices) the spot center moves enough as we traverse the z-stack
        that a larger neighborhood (~2 :math:`\mu m` seems to work fine on our system)
        should be used.
    """
    # Extract spot and background pixel values
    spot, background = extract_spot_shell(
        raw_spot,
        centroid=centroid,
        mppYX=mppYX,
        mppZ=mppZ,
        ball_diameter_um=ball_diameter_um,
        shell_width_um=shell_width_um,
        aspect_ratio=aspect_ratio,
    )

    # Generate bootstrap resamples
    spot_bootstrap = np.random.default_rng().choice(
        spot, size=(num_bootstraps, *spot.shape)
    )
    background_bootstrap = np.random.default_rng().choice(
        background, size=(num_bootstraps, *background.shape)
    )

    # Estimate intensity and error
    num_spot_pixels = spot.size

    spot_sum_bootstrap = np.sum(spot_bootstrap, axis=1)
    mean_background_bootstrap = np.mean(background_bootstrap, axis=1)

    background_intensity = num_spot_pixels * mean_background_bootstrap

    intensity_bootstrap = (
        spot_sum_bootstrap - num_spot_pixels * mean_background_bootstrap
    )

    intensity = intensity_bootstrap.mean()
    background_intensity = background_intensity.mean()
    intensity_err = intensity_bootstrap.std()

    return intensity, intensity_err, background_intensity


def extract_spot_mask(
    raw_spot, *, centroid, mppZ, mppYX, ball_diameter_um, shell_width_um, aspect_ratio
):
    """
    Generates a mask corresponding an ellipsoid neighborhood around a proposed spot, and
    to a shell around that neighborhood for spot and background quantification respectively.

    :param raw_spot: Cuboidal neighborhood around a spot as extracted by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type raw_spot: np.ndarray
    :param centroid: Centroid of spot, usually obtained by Gaussian fitting.
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :return: (ball_mask, shell_mask) where `ball_mask` is a mask of the ellipsoid
        neighborhood around the spot, and `background` is a mask of the shell around the
        neighborhood.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    # Compute distance in pixel space
    indices = np.indices(raw_spot.shape, dtype=float) + 0.5
    distance_pixels = (indices.T - np.asarray(centroid)).astype(float)

    # Use scope resolution to transform to real space
    distance_pixels[..., 0] *= mppZ * aspect_ratio
    distance_pixels[..., 1:] *= mppYX
    distance_grid = np.sqrt(np.sum(distance_pixels**2, axis=3)).T

    # Extract mask for ball centered around centroid and shell around ball
    ball_mask = distance_grid < (ball_diameter_um / 2)
    shell_mask = (distance_grid >= (ball_diameter_um / 2)) & (
        distance_grid < (ball_diameter_um / 2) + shell_width_um
    )

    return ball_mask, shell_mask


def create_blocks(
    raw_spot,
    spot_mask,
    *,
    centroid,
):
    """
    Creates a list of masks corresponding to blocks of the raw spot that get sampled
    for bootstrapping - this makes use of cylindrical symmetry of PSFs, and may not
    be applicable for non-scanning microscopy.

    :param spot_mask: Cuboidal neighborhood around a spot as extracted by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type raw_spot: np.ndarray
    :param centroid: Centroid of spot, usually obtained by Gaussian fitting.
    :type centroid: np.ndarray
    :return: list
    """
    # We iterate over xy planes and pick out pixel-thick rings to use as IID
    # sample blocks for bootstrapping. We first define an xy-distance array
    # to be used for picking out the rings in the following loop.
    xy_indices = np.indices(raw_spot.shape[1:], dtype=float) + 0.5
    xy_distance_pixels = (xy_indices.T - np.asarray(centroid[1:])).astype(float)
    xy_distance_grid = np.sqrt(np.sum(xy_distance_pixels**2, axis=2)).T

    blocks = []
    # Index over z-planes
    for i in range(raw_spot.shape[0]):
        raw_spot_xy_plane = raw_spot[i]
        spot_mask_xy_plane = spot_mask[i]

        # Iterate over pixel-thick rings
        radius = 1

        while True:
            ring_mask = (xy_distance_grid < radius) & (xy_distance_grid > (radius - 1))
            block_mask = ring_mask & spot_mask_xy_plane

            if block_mask.sum() == 0:
                break

            block = raw_spot_xy_plane[block_mask]
            blocks.append(block)
            radius += 1

    return blocks


def bootstrap_intensity(
    raw_spot,
    *,
    centroid,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
    num_bootstraps=1000,
    background="mean",
    background_pixels=None,
    background_pixels_weights=None,
):
    """
    Extracts pixel values within an ellipsoid neighborhood around a proposed spot, and
    pixel values in a shell around that neighborhood for spot and background
    quantification respectively, then estimates the intensity as the sum of spot
    pixel values in the ellipsoid mask, background-subtracted by estimating the background
    intensity using the shell around the ellipsoid spot mask. This procedure is bootstrapped
    and used to estimate the error on the intensity. Note that the bootstrapping procedure
    divides the spot into rings around the center before bootstrapping across each ring so
    that the counts on each bootstrapping block are iid by cylindrical symmetry of the PSF -
    this may not be appropriate for non-scanning microscopy images.

    :param raw_spot: Cuboidal neighborhood around a spot as extracted by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type raw_spot: np.ndarray
    :param centroid: Centroid of spot, usually obtained by Gaussian fitting.
    :type centroid: np.ndarray
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the  xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :param background: Choose whether the background returned is the mean background
        intensity per pixel or the total background subtracted over the spot. `None`
        disables background subtraction altogether.
    :type background: {"mean", "total", `None`}
    :param background_pixels: Array of background pixels used for background estimation.
        If `None` (default), the background will be estimated from a shell around the
        spot in each frame.
    :type background_pixels: np.ndarray
    :param background_pixels_weights: Weight to apply to the background pixels in the
        mean background intensity per pixel estimation procedure. This must be of the
        same shape as `background_pixels`.
    :type background_pixels_weights: np.ndarray
    :return: (intensity, intensity_err) where `intensity` is the sum of pixel values
        inside the ellipsoid spot mask, background-subtracted by estimating the average
        background intensity per pixel from the shell around the ellipsoid mask, and
        averaged over `num_bootstraps` bootstrap samples. `intensity_err` is the standard
        deviation of the same.
    :rtype: tuple

    .. note:: If the imaging settings are fast relative to the diffusion time of
        transcriptional loci, a neighborhood of ~3 sigmas is sufficient to obtain
        good quantification of the spot. Otherwise (as they are on our system, with
        ~0.6s between z-slices) the spot center moves enough as we traverse the z-stack
        that a larger neighborhood (~2 um seems to work fine on our system) should
        be used.
    """
    # Extract spot and background pixel values
    spot_mask, background_mask = extract_spot_mask(
        raw_spot,
        centroid=centroid,
        mppYX=mppYX,
        mppZ=mppZ,
        ball_diameter_um=ball_diameter_um,
        shell_width_um=shell_width_um,
        aspect_ratio=aspect_ratio,
    )

    # Generate cylindrically-symmetric blocks for bootstrapping
    blocks = create_blocks(raw_spot, spot_mask, centroid=centroid)

    # Generate bootstrap resamples
    spot_bootstrap_samples = np.zeros((spot_mask.sum(), num_bootstraps), dtype=float)
    block_samples_start_index = 0
    for block in blocks:
        block_bootstrap_samples = np.random.choice(
            block, size=(block.size, num_bootstraps), replace=True
        )
        block_samples_end_index = block_samples_start_index + block.size
        spot_bootstrap_samples[block_samples_start_index:block_samples_end_index] = (
            block_bootstrap_samples
        )
        block_samples_start_index = block_samples_end_index

    spot_bootstrap = spot_bootstrap_samples.T

    if background is not None:
        if background_pixels is None:
            background_pixels = raw_spot[background_mask]

        # Make sure weights are normalized to a probability
        if background_pixels_weights is not None:
            background_pixels_weights /= background_pixels_weights.sum()

        background_bootstrap = np.random.choice(
            background_pixels,
            size=(num_bootstraps, background_pixels.size),
            p=background_pixels_weights,
        )

        mean_background_bootstrap = np.mean(background_bootstrap, axis=1)

    else:
        mean_background_bootstrap = 0

    # Estimate intensity and error
    num_spot_pixels = spot_mask.sum()

    spot_sum_bootstrap = np.sum(spot_bootstrap, axis=1)

    background_intensity = num_spot_pixels * mean_background_bootstrap

    intensity_bootstrap = (
        spot_sum_bootstrap - num_spot_pixels * mean_background_bootstrap
    )

    intensity = intensity_bootstrap.mean()
    intensity_err = intensity_bootstrap.std()

    if background == "mean":
        background_intensity = mean_background_bootstrap.mean()
        background_intensity_err = mean_background_bootstrap.std()
    elif background == "total":
        background_intensity = background_intensity.mean()
        background_intensity_err = background_intensity.std()
    elif background is None:
        background_intensity_err = 0
    else:
        raise ValueError("`background` option must be 'mean', 'total' or `None`.")

    return intensity, intensity_err, background_intensity, background_intensity_err


def _add_neighborhood_intensity_row(
    spot_row,
    *,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
    num_bootstraps=1000,
    background="mean",
):
    """
    Performs a bootstrap estimate of the spot intensity in an ellipsoid neighborhood,
    background subtracting using a shell around the neighborhood for a row of
    the spot dataframe after gaussian fitting by `add_fits_spots_dataframe`.
    """
    raw_spot = spot_row["raw_spot"]
    centroid = spot_row[["z", "y", "x"]] - spot_row["coordinates_start"][1:]

    if raw_spot is not None:
        intensity, intensity_err, background_intensity, background_intensity_err = (
            bootstrap_intensity(
                raw_spot,
                centroid=centroid,
                mppYX=mppYX,
                mppZ=mppZ,
                ball_diameter_um=ball_diameter_um,
                shell_width_um=shell_width_um,
                aspect_ratio=aspect_ratio,
                num_bootstraps=num_bootstraps,
                background=background,
            )
        )
    else:
        intensity = np.nan
        intensity_err = np.nan
        background_intensity = np.nan
        background_intensity_err = np.nan

    return intensity, intensity_err, background_intensity, background_intensity_err


def add_neighborhood_intensity_spot_dataframe(
    spot_df,
    *,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
    num_bootstraps=1000,
    background="mean",
    inplace=True,
):
    """
    Extracts an ellipsoid neighborhood around each proposed spot in the input spot
    dataframe (preferably after refining spot centers using gaussian fitting), summing
    over the neighborhood to find the raw integrated spot intensity. A shell of pixel
    values around the neighborhood is also extracted to estimate the background for
    background subtraction. This procedure is bootstrapped to obtain an estimate of the
    error in integrated fluorescence.

    * "intensity_from_neighborhood": Estimated spot intensity by using sum of pixel
      values in ellipsoid neighborhood around spot, background-subtracted by using
      pixel values in shell around ellipsoid neighborhood to estimate background per
      pixel.
    * "intensity_std_error_from_neighborhood": Estimated error in spot intensity by
      bootstrapping the estimator.

    :param spot_df: DataFrame containing information about putative spots as output by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_df: pandas DataFrame
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the  xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :param background: Choose whether the background returned is the mean background
        intensity per pixel or the total background subtracted over the spot.
    :type background: {"mean", "total"}
    :param bool inplace: If True, the input `spot_df` is modified in-place to add the
        required columns and returns `None`. Otherwise, a modified copy is returned.
    :return: If `inplace=False`, returns copy of input `spot_df` with added columns
        for spot intensity and associated error. Otherwise returns `None`.
    :rtype: {pandas DataFrame, None}

    .. note::
        If the imaging settings are fast relative to the diffusion time of
        transcriptional loci, a neighborhood of ~3 sigmas is sufficient to obtain
        good quantification of the spot. Otherwise (as they are on our system, with
        ~0.6s between z-slices) the spot center moves enough as we traverse the z-stack
        that a larger neighborhood (~2 um seems to work fine on our system) should
        be used.
    """
    if inplace:
        spot_dataframe = spot_df
        out = None
    else:
        spot_dataframe = spot_df.copy()
        out = spot_dataframe

    bootstrap_intensity_row_func = partial(
        _add_neighborhood_intensity_row,
        mppZ=mppZ,
        mppYX=mppYX,
        ball_diameter_um=ball_diameter_um,
        shell_width_um=shell_width_um,
        aspect_ratio=aspect_ratio,
        num_bootstraps=num_bootstraps,
        background=background,
    )

    spot_dataframe[
        [
            "intensity_from_neighborhood",
            "intensity_std_error_from_neighborhood",
            "background_intensity_from_neighborhood",
            "background_intensity_std_error_from_neighborhood",
        ]
    ] = spot_dataframe.apply(bootstrap_intensity_row_func, axis=1, result_type="expand")

    return out


def add_neighborhood_intensity_spot_dataframe_parallel(
    spot_dataframe,
    *,
    mppZ,
    mppYX,
    ball_diameter_um,
    shell_width_um,
    aspect_ratio,
    client,
    num_bootstraps=1000,
    background="mean",
    evaluate=True,
    inplace=True,
):
    """
    Parallelizes `add_neighborhood_intensity_spot_dataframe` across a Dask cluster.

    :param spot_dataframe: DataFrame containing information about proposed spots as output by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_dataframe: pandas DataFrame
    :param float mppZ: Microns per pixel in z.
    :param float mppYX: Microns per pixel in the xy plane, assumed to be symmetrical.
    :param float ball_diameter_um: Diameter of ellipsoid neighborhood in the  xy plane.
    :param float shell_width_um: Width of shell to extract around the ellipsoid mask
        used to extract the spot. This is used to estimate the background. This should
        be at least a little over `mppZ` to ensure a continuous shell is extracted.
    :param float aspect_ratio: Ratio of diameter of ellipsoid neighborhood in xy to
        the diameter in z. This should be matched to the ratio of the standard deviations
        of the gaussian approximation to the PSF of the microscope in microns - for our
        system, this is empirically very close to 0.5.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param int num_bootstraps: Number of bootstrap samples of the same shape as the
        extracted pixel values to generate for intensity estimation.
    :param background: Choose whether the background returned is the mean background
        intensity per pixel or the total background subtracted over the spot.
    :type background: {"mean", "total"}
    :param bool evaluate: If True, returns a fully-evaluated modified copy of the input
        `spot_dataframe` with the required columns added. Otherwise, returns a pointer
        to a Dask task that can be evaluated and returned on demand using the `compute`
        method. Note that `inplace=True` forces evaluation regardless of this parameter.
    :param bool inplace: If True, the input `spot_df` is modified in-place to add the
        required columns and returns `None`. Otherwise, a modified copy is returned.
    :return: If `inplace=False`, returns copy of input `spot_df` with added columns
        for spot intensity and associated error. Otherwise returns `None`.
    :rtype: {pandas DataFrame, None}

    .. note:: If the imaging settings are fast relative to the diffusion time of
        transcriptional loci, a neighborhood of ~3 sigmas is sufficient to obtain
        good quantification of the spot. Otherwise (as they are on our system, with
        ~0.6s between z-slices) the spot center moves enough as we traverse the z-stack
        that a larger neighborhood (~2 um seems to work fine on our system) should
        be used.
    """
    # Preallocate columns to facilitate sharing metadata with Dask
    spot_dataframe[
        [
            "intensity_from_neighborhood",
            "intensity_std_error_from_neighborhood",
            "background_intensity_from_neighborhood",
        ]
    ] = np.nan
    num_processes = len(client.scheduler_info()["workers"])
    spot_dataframe_dask = dd.from_pandas(spot_dataframe, npartitions=num_processes)
    add_intensity_spot_func = partial(
        add_neighborhood_intensity_spot_dataframe,
        mppZ=mppZ,
        mppYX=mppYX,
        ball_diameter_um=ball_diameter_um,
        shell_width_um=shell_width_um,
        aspect_ratio=aspect_ratio,
        num_bootstraps=num_bootstraps,
        background=background,
        inplace=False,
    )

    # Map across partitions
    spot_dataframe_dask = spot_dataframe_dask.map_partitions(
        add_intensity_spot_func, meta=spot_dataframe_dask
    )

    if evaluate:
        spot_dataframe_dask = client.compute(spot_dataframe_dask)

    if inplace:
        spot_dataframe[:] = spot_dataframe_dask
        out = None
    else:
        out = spot_dataframe_dask

    return out
