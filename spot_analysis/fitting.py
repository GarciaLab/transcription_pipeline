from utils import neighborhood_manipulation
from scipy.optimize import least_squares
import numpy as np


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
    **kwargs
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
    error_out = (np.full(3, np.nan),) + (np.nan,)*5
    
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
            out = (np.full(3, np.nan),) + (np.nan,)*5
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