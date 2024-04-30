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
    **kwargs,
):
    # Check to make sure we have a consolidated array - if we have a single trace, we
    # have to wrap it in a new axis for the helper function `_corr_traces` to work.
    if traces_array.ndim == 1:
        traces_array = np.expand_dims(traces_array, axis=0)

    if (corr_type == "pearson") & (subtract_mean is None):
        warnings.warn("Pearson correlation used with `subtract_mean=False`.")

    num_timepoints = traces_array.shape[1]
    delta_t_array = np.arange(1, num_timepoints - 2)
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
                **kwargs,
            )

    return corr_array


def mean_corr_traces(
    traces_array,
    corr_traces_array=None,
    corr_type="pearson",
    subtract_mean="rolling",
    mean_window=5,
    **kwargs,
):
    corr = corr_traces(
        traces_array,
        corr_traces_array=corr_traces_array,
        corr_type=corr_type,
        subtract_mean=subtract_mean,
        mean_window=mean_window,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

        mean_corr = np.nanmean(corr, axis=0)
        stderr_corr = np.nanstd(corr, axis=0) / np.sqrt(~np.isnan(corr).sum(axis=0))

    return mean_corr, stderr_corr
