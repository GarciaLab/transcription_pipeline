import numpy as np
import dask
from dask.distributed import LocalCluster, Client


def _check_input_form(movies_list, client):
    """
    Handles possible input types for parallelized functions (list of futures or
    fully evaluated arrays) and pre-processes them for direct execution by
    parallelized functions, passing lists of futures untouched and splitting and
    scattering numpy arrays across the workers.
    """
    scattered_data = []
    for movie in movies_list:
        num_processes = len(client.scheduler_info()["workers"])

        if isinstance(movie, list):
            # Check that any already chunked and scattered input matches the
            # number of processes on the cluster
            if not len(movie) == num_processes:
                raise Exception(
                    "Inputs passed as list of Futures must match the number of processes."
                )

            if all(
                isinstance(chunk, dask.distributed.client.Future) for chunk in movie
            ):
                scattered_data.append(movie)

            else:
                raise TypeError(
                    "Unsupported input movie type, must be ndarray or list of futures."
                )

        else:
            scattered_data.append(client.scatter(np.array_split(movie, num_processes)))

    return scattered_data


def _compute_futures(scattered_data, func, client, evaluate, futures_in, futures_out):
    """
    Dispatches the data in movies_list, with each movie chunked so that each worker
    gets one chunk of each of the movies in movies_list. This data is then operated
    on using func by the workers.
    """
    processed_movie_futures = client.map(func, *scattered_data)

    if evaluate:
        results = client.gather(processed_movie_futures)
        processed_movie = np.concatenate(results)
    else:
        processed_movie = None

    if not futures_in:
        scattered_data = None

    if not futures_out:
        processed_movie_futures = None

    return processed_movie, processed_movie_futures, scattered_data


def parallelize(
    movies_list, func, client, *, evaluate=True, futures_in=True, futures_out=True
):
    """
    Parallelizes the execution of input function `func` taking elements of movies_list
    as only positional arguments across workers, dispatching the input data to
    distributed memory first if necessary.

    :param list movies_list: List with each element corresponding to a positional
        input of `func`. Each element may be an `ndarray` which is then dispatched
        to the distributed memory, or a list of pointers to Futures in the
        distributed memory that are used as is.
    :param function func: Function taking the elements of movies_list as positional
        arguments.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param bool evaluate: If True (default), deserialize and concatenate results of
        worker computations on execution. If False, processed_movie is set to None.
        This is useful for avoiding unnecessary gathering and deserialization if you
        don't need the result of a function computation by itself, but require it as
        an input to a downstream parallelized function.
    :param bool futures_in: If True (default), keeps the futures objects used to
        send the input movies from movies_list and passes them to the scattered_data
        output. If False, scattered_output is set to None and the futures objects
        are removed so that the worker memory can be garbage collected.
    :param bool futures_out: If True (default), keeps the futures objects corresponding
        to the output of the computation before concatenation and passes them to the
        processed_movie_futures output. If False, scattered_output is set to None and
        the futures objects are removed so that the worker memory can be garbage
        collected.
    :return: Tuple(processed_movie, processed_movie_futures, scattered_data) where
        *processed_movie is the fully evaluated result of the computation
        *processed_movie_futures is the list of futures objects resulting from the
        computation before concatenation.
        *scattered_data is a list of list of futures pointing to the input data in
        movies_list in the workers' memory
    :rtype: tuple
    """
    if not isinstance(client, dask.distributed.client.Client):
        raise TypeError("Provided `client` argument is not a Dask client.")

    scattered_data = _check_input_form(movies_list, client)

    processed_movie, scattered_data, processed_movie_futures = _compute_futures(
        scattered_data,
        func,
        client,
        evaluate,
        futures_in,
        futures_out,
    )

    return processed_movie, scattered_data, processed_movie_futures


def parse_parallelize_kwargs(kwargs):
    """
    Parses optional kwargs dict for relevant :func:`~parallelize` arguments.

    :param dict kwargs: Dictionary of keyword arguments.
    :return: Tuple(evaluate, futures_in, futures_out) corresponding to the values
        of the optional keyword arguments for :func:`~parallelize` with the same
        defaults (all True).
    :rtype Tuple of booleans
    """
    try:
        evaluate = kwargs["evaluate"]
    except KeyError:
        evaluate = True

    try:
        futures_in = kwargs["futures_in"]
    except KeyError:
        futures_in = True

    try:
        futures_out = kwargs["futures_out"]
    except KeyError:
        futures_out = True

    return evaluate, futures_in, futures_out


def number_of_frames(movie, client):
    """
    Finds number of frames in input `movie`, whether movie is passed as a Numpy
    array or a list of Futures corresponding to chunks of `movie`.

    :param movie: Input movie as passed wrapped in list to `parallelize`.
    :type movie: Numpy array or list of Futures corresponding to chunks of `movie`.
    :return: Number of frames in input movie.
    :rtype: int
    """
    if isinstance(movie, np.ndarray):
        num_frames = movie.shape[0]

    elif isinstance(movie, list):
        if all(isinstance(chunk, dask.distributed.client.Future) for chunk in movie):
            count_frames = lambda x: x.shape[0]
            frames = client.map(count_frames, movie)
            frames = np.array(client.gather(frames))
            num_frames = int(np.sum(frames))

        else:
            raise TypeError(
                "Unsupported input movie type, must be ndarray or list of futures."
            )

    else:
        raise TypeError(
            "Unsupported input movie type, must be ndarray or list of futures."
        )

    return num_frames