import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings


def _corr_traces(
    traces_array,
    delta_t,
    corr_type,
    corr_traces_array=None,
    difference=False,
    subtract_mean="full",
    mean_window=None,
    win_kwargs={},
    **kwargs,
):
    # Since cross-correlation is not necessarily even and symmetric around zero-lag,
    # we also have to account for negative-lag cross-correlation calculations.
    if (delta_t < 0) & (corr_traces_array is not None):
        traces_array, corr_traces_array = corr_traces_array, traces_array
        delta_t = np.abs(delta_t)

    if difference:
        traces_df = pd.DataFrame(traces_array, copy=True)
        diff_traces_df = traces_df.diff(axis=1)
        traces_array = diff_traces_df.to_numpy(dtype=float, na_value=np.nan)

    trimmed_traces = traces_array[..., : traces_array.shape[1] - delta_t].copy()

    if corr_traces_array is None:
        shifted_traces = traces_array[..., delta_t:].copy()
    else:
        shifted_traces = corr_traces_array[..., delta_t:].copy()

    # Mean subtraction
    if subtract_mean == "full":
        trimmed_mean = np.nanmean(trimmed_traces, axis=1, keepdims=True)
        shifted_mean = np.nanmean(shifted_traces, axis=1, keepdims=True)
    elif subtract_mean == "rolling":
        if "min_periods" not in kwargs:
            kwargs["min_periods"] = 1
        if "closed" not in kwargs:
            kwargs["closed"] = "both"

        mean_traces_array = (
            pd.DataFrame(traces_array.T, copy=True)
            .rolling(mean_window, **kwargs)
            .mean(**win_kwargs)
            .to_numpy(dtype=float, na_value=np.nan)
        ).T

        trimmed_mean = mean_traces_array[
            ..., : mean_traces_array.shape[1] - delta_t
        ].copy()
        shifted_mean = mean_traces_array[..., delta_t:].copy()

    elif subtract_mean is None:
        trimmed_mean = 0
        shifted_mean = 0
    else:
        raise ValueError("`subtract_mean` option not recognized.")

    trimmed_traces -= trimmed_mean
    shifted_traces -= shifted_mean

    if corr_type == "pearson":
        # Correlation estimate for each trace
        corr = np.nanmean(trimmed_traces * shifted_traces, axis=1)

        # Standard deviation estimation for normalization
        trimmed_traces_std = np.nanstd(trimmed_traces, axis=1)
        shifted_traces_std = np.nanstd(shifted_traces, axis=1)
        corr /= trimmed_traces_std * shifted_traces_std
    elif corr_type == "fcs":
        # Correlation estimate for each trace
        corr = np.nanmean((trimmed_traces * shifted_traces) / (trimmed_mean**2), axis=1)
    else:
        raise ValueError("Unknown `corr_type` option.")

    return corr


def corr_traces(
    traces_array,
    corr_traces_array=None,
    corr_type="pearson",
    subtract_mean="rolling",
    mean_window=5,
    win_kwargs={},
    **kwargs,
):
    """
    Calculates the correlation as a function of lag time, averaged across time, for an array
    of traces where each element of the array corresponds to a sample with fixed time
    resolution, with missing timepoints padded with `np.nan`s. If `corr_traces_array` is
    provided, a cross-correlation is computed. Otherwise, the autocorrelation is computed.

    :param traces_array: The array of traces with the 0-th axis enumerating traces, and the
        1-st axis corresponding to timepoints. Missing timepoints are padded with `np.nan`s.
        This can be compiled using :func:`~spot_analysis.compile_data.consolidate_traces`.
        Single traces can be passed as a 1D array.
    :type traces_array: np.ndarray
    :param corr_traces_array: If provided, the cross-correlation with the traces in
        `traces_array` is computed. Must be the same shape as `traces_array`, padded with `np.nan`
        as necessary.
    :type corr_traces_array: np.ndarray
    :param corr_type: The type of correlation to compute (this chooses the normalization).
        This changes the normalization of the correlation function, with `"pearson"`
        corresponding to division by the product of the standard deviation of the traces
        and `"fcs"` corresponding to division by the product of the average intensities
        (after mean subtraction).
    :type corr_type: {"pearson", "fcs"}
    :param subtract_mean: Method to use when performing mean subtraction on relevant subsets
        of the traces before computing the correlation.

        * `"rolling"` corresponds to averaging over a prescribed sliding time window `"mean_window"`.
          If rolling averaging is used, any keyword argument accepted by :func:`~pandas.DataFrame.rolling`
          will be passed on. If a `win_type` argument is specified, a `win_kwargs` dictionary may
          also need to be provided (see below).

        * `"full"` corresponds to simply averaging over the entire relevant subsets of the traces
          (i.e. the overlap after time delay).

        * `None` corresponds to performing correlation without mean subtraction.

    :type subtract_mean: {"rolling", "full", `None`}
    :param int mean_window: Number of timepoints to roll over during mean subtraction if
        `subtract_mean`="rolling"`.
    :param dict win_kwargs: If a `scipy.signal` window type is used, as can be done by passing
        on keyword arguments accepted by :func:`~pandas.DataFrame.rolling`, additional parameters
        matching the keywords specified in the Scipy window type method may need to be passed
        in this argument as a dictionary.
    :return: Tuple of array of correlation function with respect to lag time for each trace, in the same
        shape as `traces_array`, and array of the corresponding lag time.
    :rtype: tuple[np.ndarray]

    .. note::
        * If you have an array of traces that you want individual correlation functions for, it is
          better to pass the full array after padding rather than do it on individual traces in
          a loop - the padding adds some overhead, but that ends up being much less costly than
          the advantage we gain from leveraging Numpy's vectorized methods on arrays instead of
          looping in Python.

        * If you're calculating a cross-correlation, keep in mind that the cross-correlation is
          also calculated with negative lag time since the cross-correlation (unlike autocorrelation)
          is not necessarily symmetric around 0.

        * In the autocorrelation case, we ignore the 0-lag autocorrelation since that's just a sum of
          squares of the signal (with some normalization) and therefore still includes shot noise that
          immediately decorrelates at non-zero lag times, but there's no need to ignore the 0-lag
          cross-correlation since the noise in the cross-correlation is already uncorrelated (and if not,
          that's something that needs to be preserved).
    """
    # Check to make sure we have a consolidated array - if we have a single trace, we
    # have to wrap it in a new axis for the helper function `_corr_traces` to work.
    if traces_array.ndim == 1:
        traces_array = np.expand_dims(traces_array, axis=0)

    if (corr_type == "pearson") & (subtract_mean is None):
        warnings.warn("Pearson correlation used with `subtract_mean=False`.")

    num_timepoints = traces_array.shape[1]
    if corr_traces_array is None:
        delta_t_array = np.arange(1, num_timepoints - 2)
    else:
        delta_t_array = np.arange(-(num_timepoints - 1), (num_timepoints - 2))

    corr_array = np.empty((traces_array.shape[0], delta_t_array.shape[0]), dtype=float)
    corr_array[:] = np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        for i, delta_t in tqdm(
            enumerate(delta_t_array),
            desc="Calculating correlation",
            total=delta_t_array.shape[0],
        ):
            corr_array[:, i, ...] = _corr_traces(
                traces_array,
                delta_t,
                corr_type,
                corr_traces_array=corr_traces_array,
                subtract_mean=subtract_mean,
                mean_window=mean_window,
                win_kwargs=win_kwargs,
                **kwargs,
            )

    return corr_array, delta_t_array


def mean_corr_traces(
    traces_array,
    corr_traces_array=None,
    corr_type="pearson",
    subtract_mean="rolling",
    mean_window=5,
    win_kwargs={},
    **kwargs,
):
    """
    Averages correlation functions across traces as computed by :func:`corr_traces`.

    :param traces_array: The array of traces with the 0-th axis enumerating traces, and the
        1-st axis corresponding to timepoints. Missing timepoints are padded with `np.nan`s.
        This can be compiled using :func:`~spot_analysis.compile_data.consolidate_traces`.
        Single traces can be passed as a 1D array.
    :type traces_array: np.ndarray
    :param corr_traces_array: If provided, the cross-correlation with the traces in
        `traces_array` is computed.
    :type corr_traces_array: np.ndarray
    :param corr_type: The type of correlation to compute (this chooses the normalization).
        This changes the normalization of the correlation function, with `"pearson"`
        corresponding to division by the product of the standard deviation of the traces
        and `"fcs"` corresponding to division by the product of the average intensities
        (after mean subtraction).
    :type corr_type: {"pearson", "fcs"}
    :param subtract_mean: Method to use when performing mean subtraction on relevant subsets
        of the traces before computing the correlation.

        * `"rolling"` corresponds to averaging over a prescribed sliding time window `"mean_window"`.
          If rolling averaging is used, any keyword argument accepted by :func:`~pandas.DataFrame.rolling`
          will be passed on. If a `win_type` argument is specified, a `win_kwargs` dictionary may
          also need to be provided (see below).

        * `"full"` corresponds to simply averaging over the entire relevant subsets of the traces
          (i.e. the overlap after time delay).

        * `None` corresponds to performing correlation without mean subtraction.

    :type subtract_mean: {"rolling", "full", `None`}
    :param int mean_window: Number of timepoints to roll over during mean subtraction if
        `subtract_mean`="rolling"`.
    :param dict win_kwargs: If a `scipy.signal` window type is used, as can be done by passing
        on keyword arguments accepted by :func:`~pandas.DataFrame.rolling`, additional parameters
        matching the keywords specified in the Scipy window type method may need to be passed
        in this argument as a dictionary.
    :return: Arrays of mean correlation function with respect to lag time averaged over all
        traces, standard error of the same, and corresponding lag-times respectively.
    :rtype: tuple[np..ndarray]

    .. note::
        This is left here as a convenience function, but should almost never be used for
        anything quantitative unless you're 100% sure that your correlograms for each trace
        are going to be fine (e.g. on synthetic data simulated to be at steady-state)
        since it's quite easy for the mean to get skewed by a few bad correlograms, for
        instance if there's some bursty behavior. Instead, use :func:`corr_traces` and
        manually curate the correlograms before averaging in your own script.
    """
    delta_t_array, corr = corr_traces(
        traces_array,
        corr_traces_array=corr_traces_array,
        corr_type=corr_type,
        subtract_mean=subtract_mean,
        mean_window=mean_window,
        win_kwargs=win_kwargs,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

        mean_corr = np.nanmean(corr, axis=0)
        stderr_corr = np.nanstd(corr, axis=0) / np.sqrt(~np.isnan(corr).sum(axis=0))

    return mean_corr, stderr_corr, delta_t_array
