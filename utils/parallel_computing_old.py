import numpy as np
import dask.array as da
import zarr
import multiprocessing as mp
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


def _parallelization_kwargs(kwargs):
    """
    Constructs Dask Client kwarg dict for connecting to existing LocalCluster or
    creating new LocalCluster. Default cluster parameters are enforced here.
    """
    existing_client_kwargs = {}
    new_client_kwargs = {}

    try:
        existing_client_kwargs["client"] = kwargs["client"]
    except KeyError:
        existing_client_kwargs["client"] = None

    try:
        existing_client_kwargs["address"] = kwargs["address"]
        existing_client_kwargs["timeout"] = "2s"
    except KeyError:
        existing_client_kwargs["address"] = None

    try:
        num_processes = kwargs["num_processes"]
    except KeyError:
        num_processes = 4

    try:
        new_client_kwargs["memory_limit"] = kwargs["memory_limit"]
    except KeyError:
        new_client_kwargs["memory_limit"] = "4GB"

    new_client_kwargs["n_workers"] = int(min(0.9 * mp.cpu_count(), num_processes))
    new_client_kwargs["processes"] = True
    new_client_kwargs["threads_per_worker"] = 1

    return existing_client_kwargs, new_client_kwargs


def _compute(movies_list, func, dtype_out, client):
    """
    Dispatches the data in movies_list, with each movie chunked so that each worker
    gets one chunk of each of the movies in movies_list. This data is then operate
    on using func by the workers. Returns a tuple of the result of the operation
    (expected to be a single numpy array of the same shape as any of the movies
    in movies_list) and the scattered data as futures objects for later computations
    without needing to serialize again.
    """
    num_processes = len(client.scheduler_info()["workers"])

    scattered_data = []
    for movie in movies_list:
        scattered_data.append(client.scatter(np.array_split(movie, num_processes)))

    map = client.map(func, *scattered_data)
    results = client.gather(map)
    processed_movie = np.concatenate(results)

    return processed_movie, scattered_data, map


def _compute_map(movies_list, func, client_kwargs, dtype_out):
    """
    Runs a function func taking a list of identically-shaped arrays as the only
    positional argument in parallel across workers on a Dask client generated as per
    options in the client_kwargs dict. The input arrays need to be packed in
    respective order as a list before passing to this function.
    """
    try:
        client = client_kwargs["client"]
        if not isinstance(client, dask.distributed.client.Client):
            raise TypeError("Provided client is not a Dask client object.")

        processed_movie, scattered_data, map = _compute(movies_list, func, dtype_out, client)

    # If no client provided, start new client in context manager
    except KeyError as _:
        with Client(**client_kwargs) as client:
            processed_movie, scattered_data = _compute(
                movies_list, func, dtype_out, client
            )
            scattered_data = None

    return processed_movie, scattered_data, map


def send_compute_to_cluster(movies_list, func, kwargs, dtype_out):
    """
    Sends func parallel computation to existing client and LocalCluster if available,
    falls back to creating one if not. Uses kwargs dictionary passed down from parent
    function to determine parallelization parameters.

    :param list movies_list: List of movies that need to be dispatched to the
        LocalCluster in the order in which they are provided to func as positional
        arguments.
    :param function func: Function to apply to each chunk of the input movies in
        movies_list, with a list of the i-th blocks of each movie in movie_list
        being provided to func as its only positional arguments in the order in
        which they are provided in movies_list, parallelized across the LocalCluster.
    :param dict kwargs: Dictionary of all keyword arguments used to determine the
        parameters of the LocalCluster and parallelization. Accepts key-value
        `client` to specify a Dask client to send the computation to, as well as
        keyword arguments for the following Dask LocalCluster parameters in case
        creation of a new LocalCluster is needed:
        *`address`
        *`timeout`
        *`num_processes`
        *`memory_limit`
        *`n_workers`
        *`processes`
        *`threads_per_worker`
    :param dtype_out: Expected datatype of output of func.
    :type dtype_out: Numpy datatype of form np.array((), dtype).
    :return: Output of func.
    :rtype: Output dtype of func, usually a numpy array.
    """
    existing_client_kwargs, new_client_kwargs = _parallelization_kwargs(kwargs)

    # Try to connect to provided address. If no address or cannot connect, spin
    # up local cluster
    if (
        existing_client_kwargs["address"] is None
        and existing_client_kwargs["client"] is None
    ):
        func_eval = _compute_map(movies_list, func, new_client_kwargs, dtype_out)
    else:
        try:
            func_eval = _compute_map(
                movies_list, func, existing_client_kwargs, dtype_out
            )
        except OSError:
            warnings.warn(
                "".join(
                    [
                        "Cluster not found at specified address, ",
                        "starting new cluster with default parameters.",
                    ]
                ),
                stacklevel=2,
            )
            func_eval = _compute_map(movies_list, func, new_client_kwargs, dtype_out)

    processed_movie, scattered_data, map = func_eval

    return processed_movie, scattered_data, map