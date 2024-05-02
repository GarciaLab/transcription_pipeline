import numpy as np
import dask
from dask.distributed import LocalCluster


def zarr_to_futures(zarr_array, client):
    """
    Reads `zarr` array containing the image data as a list of Dask `Futures` objects
    consistent with the existing chunking on the `zarr` array.
    """
    chunk_timepoints = zarr_array.chunks[0]
    chunk_boundaries = (
        np.arange(int(np.floor(zarr_array.shape[0] / chunk_timepoints)))
        * chunk_timepoints
    )
    # if zarr_array.shape[0] % chunk_timepoints != 0:
    chunk_boundaries = np.append(chunk_boundaries, zarr_array.shape[0])

    chunk_start_stop = np.array(
        [chunk_boundaries[:-1], chunk_boundaries[1:]]
    ).T.tolist()

    zarr_futures = client.map(lambda x: zarr_array[x[0] : x[1]], chunk_start_stop)

    return chunk_start_stop, zarr_futures


def futures_to_zarr(futures, chunk_start_stop, zarr_out, client):
    """
    Indexes into a `zarr` array and transfers in values from the corresponding
    Dask `Futures` object.

    :param futures: List of futures objects containing chunks of the output, e.g. from a
        `client.map` operation.
    :type futures: List of `dask.distribution.client.Futures` objects.
    :param list chunk_start_stop: 2D list, with each 2-element along axis 0 corresponding
        to the first and last indices of the target `zarr_out` to copy values to.
    :param zarr_out: Target `zarr` array to copy values from `futures` into.
    :type zarr_out: Zarr array.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: List of `None` objects of same length as `futures`.
    """

    def _map_chunk_future_to_zarr(future, chunk_bounds):
        # This could easily be done with a lambda function, just keeping it here
        # for readability and to make the `None` return explicit.
        zarr_out[chunk_bounds[0] : chunk_bounds[1]] = future
        return None

    dummy_futures = client.map(_map_chunk_future_to_zarr, futures, chunk_start_stop)

    return client.gather(dummy_futures)


def _check_input_form(movies_list, client, *, num_chunks=None):
    """
    Handles possible input types for parallelized functions (list of futures or
    fully evaluated arrays) and pre-processes them for direct execution by
    parallelized functions, passing lists of futures untouched and splitting and
    scattering numpy arrays across the workers.
    """
    scattered_data = []
    for movie in movies_list:
        if isinstance(movie, list):
            if all(
                isinstance(chunk, dask.distributed.client.Future) for chunk in movie
            ):
                scattered_data.append(movie)

            else:
                raise TypeError(
                    "Unsupported input movie type, must be ndarray or list of futures."
                )

        else:
            num_processes = len(client.scheduler_info()["workers"])
            if num_chunks is None:
                num_chunks = num_processes
            scattered_data.append(client.scatter(np.array_split(movie, num_chunks)))

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
        del scattered_data
        scattered_data = None

    if not futures_out:
        del processed_movie_futures
        processed_movie_futures = None

    return processed_movie, processed_movie_futures, scattered_data


def parallelize(
    movies_list,
    func,
    client,
    *,
    evaluate=True,
    futures_in=True,
    futures_out=True,
    num_chunks=None
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
    :param int num_chunks: Number of chunks to split input movie into if fed as Numpy
        array. Defaults to same as number of worker processes in the Dask `client`.
    :return: tuple(processed_movie, processed_movie_futures, scattered_data) where

        * processed_movie is the fully evaluated result of the computation
        * processed_movie_futures is the list of futures objects resulting from the
          computation before concatenation.
        * scattered_data is a list of list of futures pointing to the input data in
          movies_list in the workers' memory

    :rtype: tuple
    """
    if not isinstance(client, dask.distributed.client.Client):
        raise TypeError("Provided `client` argument is not a Dask client.")

    scattered_data = _check_input_form(movies_list, client, num_chunks=num_chunks)

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
    :return: tuple(evaluate, futures_in, futures_out) corresponding to the values
        of the optional keyword arguments for :func:`~parallelize` with the same
        defaults (all True).
    :rtype: tuple[bool]
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
    :type movie: {np.ndarray, list}
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :return: Number of frames in input movie.
    :rtype: int
    """
    if isinstance(movie, np.ndarray):
        num_frames = movie.shape[0]

    elif isinstance(movie, list):
        if all(isinstance(chunk, dask.distributed.client.Future) for chunk in movie):
            frames = client.map(lambda x: x.shape[0], movie)
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
