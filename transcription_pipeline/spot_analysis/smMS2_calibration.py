import numpy as np


def log_likelihood_step(
    step,
    offset,
    *,
    intensity,
    sigma_intensity,
):
    """
    Computes the log-likelihood
    """
    # We sum over steps up to 6 sigmas away from the highest intensity data
    # point.
    try:
        steps_padding = int(np.ceil(sigma_intensity.max() / step)) * 6
    except AttributeError:  # Handle `sigma_intensity` being fed as a single fixed value
        steps_padding = int(np.ceil(sigma_intensity / step)) * 6

    num_steps = int(np.ceil((intensity.max() - offset) / step)) + steps_padding

    point_step_deviation = np.expand_dims(intensity, axis=-1) - np.expand_dims(
        offset + (np.arange(1, num_steps + 1) * step), axis=0
    )

    # Handle single sigma for all emission levels vs error-dependent sigma array
    try:
        sigma_intensity = np.expand_dims(sigma_intensity, axis=1)
    except np.AxisError:
        pass

    point_step_likelihood = (1 / (sigma_intensity * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((point_step_deviation / sigma_intensity) ** 2)
    )

    # We renormalize by a factor of the step size to convert to a probability
    point_likelihood = point_step_likelihood.sum(axis=1) * step

    log_likelihood = np.log(point_likelihood).sum()

    return log_likelihood


def vectorized_log_likelihood_step(step, offset, *, intensity, sigma_intensity):
    """ """
    try:
        step_array_size = len(step)
    except TypeError:
        step_array_size = 1
        step = np.array([step])

    try:
        offset_array_size = len(offset)
    except TypeError:
        offset_array_size = 1
        offset = np.array([offset])

    likelihood_array = np.zeros((step_array_size, offset_array_size), dtype=np.float64)
    likelihood_indices = np.indices(likelihood_array.shape).T.squeeze()

    for loc in likelihood_indices:
        likelihood_array[tuple(loc)] = log_likelihood_step(
            step[loc[0]],
            offset[loc[1]],
            intensity=intensity,
            sigma_intensity=sigma_intensity,
        )

    return likelihood_array.squeeze()
