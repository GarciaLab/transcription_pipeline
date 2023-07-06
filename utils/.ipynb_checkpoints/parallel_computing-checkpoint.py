import dask.array as da
import zarr
import multiprocessing as mp
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
    creating new LocalCluster.
    """
    existing_client_kwargs = {}
    new_client_kwargs = {}

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


def _compute_map(movies_list, func, client_kwargs, dtype_out):
    """
    Runs a function func taking a number of identically-shaped dask arrays as
    positional arguments in parallel across chunks of a dask array on a Dask client
    generated as per options in the client_kwargs dict. The input dask arrays need
    to be packed in respective order as a list before passing to this function.
    """
    with Client(**client_kwargs) as client:
        num_processes = len(client.scheduler_info()["workers"])

        # Figure out how to split movie into chunks to distribute across processes
        num_timepoints_per_chunk = int(np.ceil(movies_list[0].shape[0] / num_processes))
        num_axes = len(movies_list[0].shape)
        chunk_shape = (num_timepoints_per_chunk,) + movies_list[0].shape[1:num_axes]

        dask_movies_list = []
        for movie in movies_list:
            dask_movies_list.append(_convert_to_dask(movie, chunk_shape))

        denoised_map = da.map_blocks(
            func, *dask_movies_list, meta=np.array((), dtype=dtype_out)
        )
        denoised_movie = denoised_map.compute()

    return denoised_movie


def send_compute_to_cluster(movies_list, func, kwargs, dtype_out):
    """
    Sends func parallel computation to existing LocalCluster if available, falls
    back to creating one if not. Uses kwargs dictionary passed down from parent
    function to determine parallelization parameters.

    :param list movies_list: List of movies that need to be dispatched to the
        LocalCluster in the order in which they are provided to func as positional
        arguments.
    :param function func: Function to apply to each chunk of the input movies in
        movies_list, with the i-th blocks of each movie in movie_list being 
        provided to func as positional arguments in the order in which the are
        provided in the list, parallelized across the LocalCluster.
    :param dict kwargs: Dictionary of all keyword arguments used to determine the
        parameters of the LocalCluster and parallelization. Accepts key-values for
        the following Dask LocalCluster parameters:
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
    if existing_client_kwargs["address"] is None:
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

    return func_eval