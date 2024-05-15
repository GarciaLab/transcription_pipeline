from skimage.segmentation import expand_labels
from functools import partial
from . import parallel_computing
import numpy as np


def expand_labels_movie(label_movie, *, distance=1):
    """
    Extends `skimage.segmentation.expand_labels` to operate frame-by-frame on a movie.

    :param label_movie: Labelled movie.
    :type label_movie: np.ndarray
    :param int distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :return: Labelled movie array with connected regions enlarged frame-by-frame.
    :rtype: np.ndarray
    """
    num_timepoints = label_movie.shape[0]
    expanded_labels = np.empty_like(label_movie)
    for i in range(num_timepoints):
        expanded_labels[i] = expand_labels(label_movie[i], distance=distance)

    return expanded_labels


def expand_labels_movie_parallel(label_movie, *, client, distance=1, **kwargs):
    """
    Extends `skimage.segmentation.expand_labels` to operate frame-by-frame on a movie,
    parallelized across a Dask LocalCluster.

    :param label_movie: Labelled movie.
    :type label_movie: np.ndarray
    :param int distance: Euclidean distance in pixels by which to grow the labels,
        defaults to 1.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: tuple(`expanded_labels`, `expanded_labels_futures`, `scattered_label_movie`)
        where

        * `expanded_labels` is the fully evaluated labelled movie array with connected
          regions enlarged frame-by-frame.
        * `expanded_labels_futures` is the list of futures objects resulting from the
          label expansion in the worker memories before gathering and concatenation.
        * `scattered_label_movie` is a list with each element corresponding to a list of
          futures pointing to the input `label_movie` in the workers' memory.

    :rtype: tuple

    .. note::

        This function can also pass along any kwargs taken by
        :func:`~utils.parallel_computing.parallelize`.

    """
    expand_labels_func = partial(expand_labels_movie, distance=distance)

    evaluate, futures_in, futures_out = parallel_computing.parse_parallelize_kwargs(
        kwargs
    )

    (
        expanded_labels,
        expanded_labels_futures,
        scattered_label_movie,
    ) = parallel_computing.parallelize(
        [label_movie],
        expand_labels_func,
        client,
        evaluate=evaluate,
        futures_in=futures_in,
        futures_out=futures_out,
    )

    return expanded_labels, expanded_labels_futures, scattered_label_movie
