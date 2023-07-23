from utils import neighborhood_manipulation
from scipy.optimize import least_squares
import numpy as np
from functools import partial
import dask.dataframe as dd


def gaussian3d_sym_xy(coordinates, *, centroid, sigma_x_y, sigma_z, amplitude, offset):
    """
    Evaluates a 3d Gaussian function isotropic in the xy-plane and with specified
    centroid at specified coordinates.

    :param coordinates: Coordinates at which to evaluate the Gaussian.
    :type coordinates: Array-like.
    :param centroid: Coordinates of centroid of Gaussian function to evaluate.
    :type centroid: Array-like.
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
    box_shape, centroid, sigma_x_y, sigma_z, amplitude, offset
):
    """
    Evaluates a 3d Gaussian function isotropic in the xy-plane and with specified
    centroid at specified coordinates.

    :param box_shape: Shape of bounding box containing the generated Gaussian.
    :type box_shape: Array-like.
    :param centroid: Coordinates of centroid of Gaussian function to evaluate.
    :type centroid: Array-like.
    :param float sigma_x_y: Standard deviation of Gaussian function in the x- and y-
        coordinate direction.
    :param float sigma_z: Standard deviation of Gaussian function in the z-coordinate.
    :param float amplitude: Amplitude of Gaussian function.
    :param float offset: Offset (limit away from centroid) of Gaussian.
    :return: Generated Gaussian inside bounding box.
    :rtype: Numpy array
    """
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
    :type data: Numpy array.
    :param centroid_guess: Intial guess for Gaussian centroid to feed to least
        squares minimization.
    :type centroid: Array-like.
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
    :param str method: Method to use for least-squares optimization (see
        `scipy.optimize.least_squares`).
    :return: If optimization is successful, returns a tuple of fit parameters
        `(centroid, sigma_x_y, sigma_z, amplitude, offset, cost)`. The notation
        is consistent with that used in `gaussian3d_sym_xy`.
        *`centroid`: Centroid of fitted 3D Gaussian (Numpy array).
        *`sigma_x_y`: Standard deviation of fitted 3D Gaussian in x- and y-coordinate.
        *`sigma_z`: Standard deviation of fitted 3D Gaussian in z-coordinate.
        *`amplitude`: Amplitude of fitted 3D Gaussian.
        *`offset`: Offset of fitted 3D Gaussian.
        *`cost`: Value of cost function at the solution.
    :rtype: Tuple

    .. note:: This function can also pass through any kwargs accepted by
        `scipy.optimization.least_squares`.
    """
    error_out = (np.full(3, np.nan),) + (np.nan,) * 5

    if data is None:
        out = error_out

    else:
        if amplitude_guess is None:
            amplitude_guess = data.max()

        if offset_guess is None:
            offset_guess = _extract_perimeter_mean(data)

        param_initial = np.array(
            [
                *centroid_guess,
                sigma_x_y_guess,
                sigma_z_guess,
                amplitude_guess,
                offset_guess,
            ]
        )

        residuals_func = lambda x: (
            generate_gaussian_3d_sym_xy(data.shape, [x[0], x[1], x[2]], *x[3:]) - data
        ).ravel()

        result = least_squares(
            residuals_func, param_initial, bounds=(0, np.inf), method=method, **kwargs
        )

        if not result.success:
            out = (np.full(3, np.nan),) + (np.nan,) * 5
        else:
            params = result.x
            centroid = params[:3]
            sigma_x_y = params[3]
            sigma_z = params[4]
            amplitude = params[5]
            offset = params[6]
            cost = result.cost

            out = (centroid, sigma_x_y, sigma_z, amplitude, offset, cost)

    return out


def _fit_spot_dataframe_row(
    row,
    *,
    sigma_x_y_guess,
    sigma_z_guess,
    amplitude_guess,
    offset_guess,
    method,
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
        offset_guess=amplitude_guess,
        method=method,
        **kwargs,
    )

    fit_relative_centroid = fit_results[0]

    # Restore initial centroid values if fit failed
    if np.isnan(fit_relative_centroid).any():
        fit_relative_centroid = centroid_guess

    refined_centroid = spatial_coordinates_start + fit_relative_centroid

    out = (*refined_centroid, *fit_results[1:])

    return out


def add_fits_spots_dataframe(
    spot_df,
    *,
    sigma_x_y_guess,
    sigma_z_guess,
    amplitude_guess=None,
    offset_guess=None,
    method="trf",
    inplace=True,
    **kwargs,
):
    """
    Fits a 3D Gaussian to the raw data in each row of the DataFrame of detected spots
    output by :func:`~spot_analysis.detection.detect_and_gather_spots` using
    :func:`~fit_gaussian_3d_sym_xy`. The fit centroid is then used to refine the
    spatial coordinates of the spot, and columns are added as follows:
    *"sigma_x_y": Standard deviation of Gaussian fit in x- and y-coordinates.
    *"sigma_z": Standard deviation of Gaussian fit in z-coordinate.
    *"amplitude": Amplitude of Gaussian fit (this is typically the signal of interest).
    *"offset": Offset of Gaussian fit (useful for background subtraction).
    *"cost": Value of the cost function at termination of the fit, see documentation
    for `scipy.optimize.least_squares`.
    *"norm_cost": Normalized cost, defined as the L2-norm of the residuals divided
    by the product of the amplitude and the number of voxels. This gives a dimensionless
    measure of cost of the fit that can be more easily used for downstream filtering.

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
    )
    spot_dataframe[
        ["z", "y", "x", "sigma_x_y", "sigma_z", "amplitude", "offset", "cost"]
    ] = spot_dataframe.apply(fit_spot_row_func, result_type="expand", axis=1)

    # Add normalized cost as a dimensionless metric for easier filtering based o
    # goodness-of-fit. We multiply the cost function by 2 to cancel out the factor of
    # 1/2 that `scipy.optimize.least_squares` adds to the cost (it does this to get a nicer
    # algebraic form for the gradient, but we readjust it to keep consistent with the
    # L2 norm definition).
    attempted_fit_mask = ~spot_dataframe["cost"].apply(np.isnan)
    dataset_size = spot_dataframe.loc[attempted_fit_mask, "raw_spot"].apply(
        lambda x: x.size
    )
    amplitude = spot_dataframe.loc[attempted_fit_mask, "amplitude"]
    spot_dataframe["norm_cost"] = np.nan
    spot_dataframe.loc[attempted_fit_mask, "norm_cost"] = (
        (2 * spot_dataframe.loc[attempted_fit_mask, "cost"])
        .apply(np.sqrt)
        .divide(dataset_size * amplitude)
    )

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

    :param spot_df: DataFrame containing information about putative spots as output by
        :func:`~spot_analysis.detection.detect_and_gather_spots`.
    :type spot_df: pandas DataFrame
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
        method. Note that `in_place=True` forces evaluation regardless of this parameter.
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
        ["sigma_x_y", "sigma_z", "amplitude", "offset", "cost", "norm_cost"]
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
        spot_dataframe_dask = spot_dataframe_dask.compute()
        out = spot_dataframe_dask

    if inplace:
        spot_dataframe[:] = spot_dataframe_dask
        out = None
    else:
        out = spot_dataframe_dask

    return out