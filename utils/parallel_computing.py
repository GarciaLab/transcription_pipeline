import numpy as np
import dask.array as da
import zarr
import dask
from dask.distributed import LocalCluster, Client


def _convert_to_dask(movie, chunk_shape):
    """
    Converts supported array types to dask with requested chunking.
    """
    if isinstance(movie, np.ndarray):
        dask_movie = da.from_array(movie, chunks=chunk_shape)
    elif isinstance(movie, zarr.core.Array):
        dask_movie = da.from_zarr(movie, chunks=chunk_shape)
    elif isinstance(movie, da.Array):
        dask_movie = da.rechunk(movie, chunks=chunk_shape)
    else:
        raise Exception("Movie data type not recognized, must be numpy, zarr or dask.")

    return dask_movie


def compute_futures(movies_list, func, client, futures_in, futures_out):
    """
    Dispatches the data in movies_list, with each movie chunked so that each worker
    gets one chunk of each of the movies in movies_list. This data is then operated
    on using func by the workers.
    
    :param list movies_list: List of input numpy arrays corresponding to movies being
        analyzed.
    :param function func: Function taking the elements of movies_list as positional
        arguments.
    :param client: Dask client to send the computation to.
    :type client: `dask.distributed.client.Client` object.
    :param bool futures_in: If True (default), keeps the futures objects used to
        send the input movies from movies_list and passes them to the scattered_data
        output. If False, scattered_output is set to None and the futures objects
        are removed so that the worker memory can be garbage collected.
    :param bool futures_out: If True (default), keeps the futures objects corresponding
        to the output of the computation before concatenation and passes them to the
        processed_movie_futures output. If False, scattered_output is set to None and
        the futures objects are removed so that the worker memory can be garbage
        collected.
    :return: Tuple(processed_movie, scattered_data, processed_movie_futures) where
        *processed_movie is the fully evaluated result of the computation
        *scattered_data is a list of list of futures pointing to the input data in
        movies_list in the workers' memory
        *processed_movie_futures is the list of futures objects resulting from the
        computation before concatenation.
    :rtype: tuple
    """
    if not isinstance(client, dask.distributed.client.Client):
        raise TypeError('Provided client argument is not a Dask client.')
        
    num_processes = len(client.scheduler_info()["workers"])

    scattered_data = []
    for movie in movies_list:
        scattered_data.append(client.scatter(np.array_split(movie, num_processes)))

    processed_movie_futures = client.map(func, *scattered_data)
    results = client.gather(processed_movie_futures)
    processed_movie = np.concatenate(results)

    if not futures_in:
        scattered_data = None

    if not futures_out:
        processed_movie_futures = None

    return processed_movie, scattered_data, processed_movie_futures