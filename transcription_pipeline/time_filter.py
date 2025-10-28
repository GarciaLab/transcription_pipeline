import os
import zarr
import numpy as np
from tqdm import tqdm
import warnings
from glob import glob
import shutil


def temporal_filter_zarr(
        zarr_arr,
        window,
        *,
        mode="mean",
        weights=None,
        out=None,
        chunk_len=None,
        monitor_progress=True,
        use_gpu=True,
    ):
    """
    Sliding temporal mean filter along axis=0 for a zarr array (GPU with CuPy).
    
    Parameters
    ----------
    zarr_arr : zarr.Array
        Input 4D array (t, z, y, x). Must be chunked along time.
    window : int
        Window size (must be odd).
    mode : str
        Whether to 'mean' filter or 'median' filter (both using provided weight
        kernel).
    weights : cp.ndarray or np.ndarray, optional
        1D Weight kernel (centered at 0) to use in temporal mean filter.
        Must be of length `window`, default is equivalent to `cp.ones(window)`.
    out : zarr.Array or np.ndarray
        Preallocated output array, same shape as zarr_arr.
    chunk_len : int, optional
        Length of time-chunk to load. Defaults to zarr_arr.chunks[0]. If the
        input `zarr_arr` is already chunked with a chunk size too small
        (`NotImplemented` error), force a larger chunk size here, ideally an
        integer multiple of the data's chunk size.
    monitor_progress : bool
        Show tqdm progress bar.
    use_gpu : bool
        If `True`, will use CuPy instead of Numpy if available.
    
    Notes
    -----
    - Requires zarr_arr.chunks[0] >= window.
    - Handles boundaries with nearest-padding behavior.
    - Keeps only 2 adjacent chunks in GPU memory at once.
    """
    weights, chunk_len, cp, tdim, half, pbar = checks(
        zarr_arr,
        window, 
        weights, 
        chunk_len, 
        monitor_progress, 
        use_gpu
    )

    # preload first chunk
    (
        chunk0_start,
        chunk0_end, 
        chunk0, 
        chunk1_start, 
        chunk1_end, 
        chunk1, 
        first_frame, 
        last_frame, 
        init_block
    ) = load_init_chunk(zarr_arr, chunk_len, cp, tdim, half)

    # Preallocate array for weighted mean if necessary
    (
        out, 
        window_array, 
        buffer, 
        buf_ptr, 
        chunk_start, 
        current_mean
    ) = preallocate_buffers(
        zarr_arr, 
        window, 
        weights, 
        out, 
        chunk_len, 
        cp, 
        init_block,
        mode,
    )
    if pbar is not None:
        pbar.update(1)

    # helper to fetch frame from current chunks
    def get_frame(
            idx,
            chunk0_start,
            chunk0_end,
            chunk0,
            chunk1_start,
            chunk1_end,
            chunk1,
        ):
        if chunk0 is not None and chunk0_start <= idx < chunk0_end:
            return chunk0[idx - chunk0_start]
        elif chunk1 is not None and chunk1_start <= idx < chunk1_end:
            return chunk1[idx - chunk1_start]
        elif idx < 0:
            return first_frame
        elif idx >= tdim:
            return last_frame
        else:
            raise RuntimeError(f"Index {idx} outside loaded chunks")

    # main loop
    for t in range(1, tdim):
        prev_idx = max(0, t - half - 1)
        next_idx = min(tdim - 1, t + half)

        # proactive swap for next_idx (if window extends beyond chunk0)
        (
            window_array,
            next_frame,
            prev_frame,
            chunk0,
            chunk0_start,
            chunk0_end,
            chunk1,
            chunk1_start,
            chunk1_end,
        ) = index_buffers(
            zarr_arr, 
            weights, 
            chunk_len, 
            cp, 
            tdim,
            chunk0_start,
            chunk0_end,
            chunk0,
            chunk1_start, 
            chunk1_end, 
            chunk1, 
            window_array, 
            get_frame, 
            prev_idx, 
            next_idx
        )

        # incremental update
        if window_array is None and mode == "mean":
            current_mean += (next_frame - prev_frame) / window
        elif mode == "mean":
            current_mean = cp.sum(window_array * weights, axis=0)
        elif mode == "median":
            current_mean = weighted_median(
                window_array,
                weights=weights,
                arr_module=cp,
            )
        else:
            raise ValueError("`mode` not recognized.")

        buffer[buf_ptr] = current_mean
        buf_ptr += 1

        # flush full buffer or final partial buffer
        if buf_ptr == chunk_len or t == tdim - 1:
            if isinstance(out, np.ndarray) or isinstance(out, zarr.Array):
                try:
                    out[chunk_start:chunk_start + buf_ptr] = cp.asnumpy( # type: ignore
                        buffer[:buf_ptr]
                    )
                except AttributeError:
                    out[chunk_start:chunk_start + buf_ptr] = buffer[:buf_ptr]
            elif isinstance(out, cp.ndarray):
                out[chunk_start:chunk_start + buf_ptr] = buffer[:buf_ptr]
            else:
                raise ValueError(f"`out` must be a Numpy, CuPy or Zarr array, not {type(out)}")

            chunk_start += buf_ptr
            buf_ptr = 0

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return out

def index_buffers(
        zarr_arr, 
        weights, 
        chunk_len, 
        cp, 
        tdim,
        chunk0_start,
        chunk0_end,
        chunk0,
        chunk1_start, 
        chunk1_end, 
        chunk1, 
        window_array, 
        get_frame, 
        prev_idx, 
        next_idx
    ):
    if next_idx >= chunk1_end:
        chunk0, chunk0_start, chunk0_end = chunk1, chunk1_start, chunk1_end
        next_chunk_start = chunk0_end
        next_chunk_end = min(next_chunk_start + chunk_len, tdim)
        chunk1 = (
                cp.asarray(zarr_arr[next_chunk_start:next_chunk_end])
                if next_chunk_start < tdim else None
            )
        chunk1_start, chunk1_end = next_chunk_start, next_chunk_end

        # safe frame fetches
    next_frame = get_frame(
        next_idx,
        chunk0_start,
        chunk0_end,
        chunk0,
        chunk1_start,
        chunk1_end,
        chunk1,
    )
    if weights is None:
        prev_frame = get_frame(
            prev_idx,
            chunk0_start,
            chunk0_end,
            chunk0,
            chunk1_start,
            chunk1_end,
            chunk1,
        )
    else:
        prev_frame = None

        # If doing weighted average, roll preallocated array
    if window_array is not None:
        idx = (cp.arange(window_array.shape[0]) + 1) % window_array.shape[0]
        window_array = window_array[idx]
        window_array[-1] = next_frame
    return (
        window_array,
        next_frame,
        prev_frame,
        chunk0,
        chunk0_start,
        chunk0_end,
        chunk1,
        chunk1_start,
        chunk1_end
    )

def preallocate_buffers(
        zarr_arr,
        window,
        weights,
        out,
        chunk_len,
        cp,
        init_block,
        mode
    ):
    if weights is not None or mode == "median":
        window_array = init_block[:window].copy()
    else:
        window_array = None

    if mode == "mean":
        if weights is None:
            current_mean = cp.mean(init_block[:window], axis=0)
        else:
            current_mean = cp.sum(window_array * weights, axis=0)
    elif mode == "median":
        current_mean = weighted_median(
            window_array,
            weights=weights,
            arr_module=cp,
        )
    else:
        raise ValueError("`mode` not recognized, must be `mean` or `median`.")

    if out is None:
        out = cp.empty(zarr_arr.shape, dtype=cp.float32)

    # gpu buffer for chunk-wise flush
    buffer = cp.empty((chunk_len,) + zarr_arr.shape[1:], dtype=cp.float32)
    buf_ptr = 0
    chunk_start = 0

    # write first result
    buffer[buf_ptr] = current_mean
    buf_ptr += 1
    return out,window_array,buffer,buf_ptr,chunk_start,current_mean

def checks(zarr_arr, window, weights, chunk_len, monitor_progress, use_gpu):
    if use_gpu:
        try:
            import cupy as cp
            _GPU = True
        except ImportError:
            warnings.warn("Cannot import CuPy, defaulting to Numpy (CPU).")
            import numpy as cp
            _GPU = False
    else:
        import numpy as cp
        _GPU = False

    if window % 2 == 0:
        raise ValueError("Window must be odd")
    
    if chunk_len is None:
        chunk_len = zarr_arr.chunks[0]

    if chunk_len < window:
        raise NotImplementedError("Chunk size in time smaller than window")
    
    if weights is not None:
        weights = cp.asarray(weights, dtype=cp.float32)
        weights /= weights.sum()
        weights = cp.expand_dims(
            weights,
            axis=tuple(range(1, zarr_arr.ndim))
        )

    tdim = zarr_arr.shape[0]
    half = window // 2

    if monitor_progress:
        pbar = tqdm(total=tdim, desc="Mean filtering")
    else:
        pbar = None
    return weights,chunk_len,cp,tdim,half,pbar

def load_init_chunk(zarr_arr, chunk_len, cp, tdim, half):
    chunk0_start = 0
    chunk0_end = min(chunk_len, tdim)
    chunk0 = cp.asarray(zarr_arr[chunk0_start:chunk0_end])

    # preload second chunk if available
    chunk1_start = chunk0_end
    chunk1_end = min(chunk1_start + chunk_len, tdim)
    chunk1 = (
        cp.asarray(zarr_arr[chunk1_start:chunk1_end])
        if chunk1_start < tdim else None
    )

    # boundary frames for nearest padding
    first_frame = cp.asarray(zarr_arr[0])
    last_frame = cp.asarray(zarr_arr[-1])

    # initial block with left padding
    pad_block = cp.repeat(first_frame[None], half, axis=0)
    init_block = cp.concatenate([pad_block, chunk0], axis=0)
    return (
        chunk0_start,
        chunk0_end,
        chunk0,
        chunk1_start,
        chunk1_end,
        chunk1,
        first_frame,
        last_frame,
        init_block
    )

def weighted_median(values, weights, arr_module=np):
    if weights is None:
        values_copy = values.copy() # This bounds the memory footprint by allowing overwriting
        return arr_module.median(values_copy, overwrite_input=True, axis=0)
    else:
        order = arr_module.empty_like(values, dtype=int)
        for i in range(values.shape[1]):          # loop over Z
            order[:, i,...] = arr_module.argsort(values[:, i,...], axis=0)

        reordered_weights = arr_module.take_along_axis(weights, order, axis=0)
        reordered_values = arr_module.take_along_axis(values, order, axis=0)
        del order

        arr_module.cumsum(reordered_weights, axis=0, out=reordered_weights)
        med_mask = reordered_weights > 0.5

        del reordered_weights
        med_idx = arr_module.argmax(med_mask, axis=0, keepdims=True)

        out = arr_module.take_along_axis(reordered_values, med_idx, axis=0)[0]
        del reordered_values

        return out

def process_embryo_time_filter(embryo_rel_path, dataset_folder, mode = "mean", time_chunk=50, temporal_window=21):
    """
    Rechunk and apply temporal mean filtering to all collated_dataset*.zarr files in the embryo folder.
    Skips rechunking or filtering if output files already exist.

    Parameters:
        embryo_rel_path (str): Path relative to dataset_folder for the embryo
        dataset_folder (str): Base dataset folder
        time_chunk (int): Chunk size along time axis for rechunking
        temporal_window (int): Window size for temporal mean filter
    """
    embryo_full_path = os.path.join(dataset_folder, embryo_rel_path, "collated_dataset")

    # Find all collated_dataset*.zarr files
    zarr_files = glob(os.path.join(embryo_full_path, "collated_dataset*.zarr"))

    if not zarr_files:
        print(f"\n[WARNING] No collated_dataset*.zarr files found in:\n  {embryo_full_path}\n")
        return

    print(f"\n{'=' * 70}")
    print(f"PROCESSING TIME FILTER")
    print(f"{'=' * 70}")
    print(f"Embryo path: {embryo_rel_path}")
    print(f"Found {len(zarr_files)} zarr file(s) to process\n")

    for z_file in zarr_files:
        base_name = os.path.basename(z_file)
        
        # Skip if this is already a rechunked or filtered file
        if "_rechunk" in base_name or "_time_filtered" in base_name:
            continue
            
        rechunk_store = os.path.join(
            embryo_full_path,
            base_name.replace(".zarr", "_rechunk.zarr")
        )
        filtered_store = os.path.join(
            embryo_full_path,
            base_name.replace(".zarr", "_time_filtered.zarr")
        )

        print(f"\n{'-' * 70}")
        print(f"Processing: {base_name}")
        print(f"{'-' * 70}")

        # Skip if already filtered
        if os.path.exists(filtered_store):
            print(f"  ✓ Filtered file already exists - skipping all processing")
            continue

        # ------------------------------
        # Rechunk
        # ------------------------------
        if not os.path.exists(rechunk_store):
            print(f"  → Rechunking data (chunk size: {time_chunk})...")
            os.makedirs(os.path.dirname(rechunk_store), exist_ok=True)
            img = zarr.open(z_file, mode='r')
            chunks = (time_chunk, *img.shape[1:])
            dst_rechunk = zarr.open_array(
                store=rechunk_store,
                mode="w",
                shape=img.shape,
                chunks=chunks,
                dtype=np.float32,
            )
            dst_rechunk[:] = img[:].astype(np.float32)
            os.sync()
            print(f"  ✓ Rechunking complete")
        else:
            print(f"  ✓ Rechunked file already exists - reusing")

        # ------------------------------
        # Temporal mean filter
        # ------------------------------
        print(f"  → Applying temporal mean filter (window: {temporal_window})...")
        os.makedirs(os.path.dirname(filtered_store), exist_ok=True)
        dst_filtered = zarr.open_array(
            store=filtered_store,
            mode="w",
            shape=zarr.open(rechunk_store, mode='r').shape,
            chunks=(time_chunk, *zarr.open(rechunk_store, mode='r').shape[1:]),
            dtype=np.float32,
        )

        temporal_filter_zarr(
            zarr.open(rechunk_store, mode="r"),
            temporal_window,
            mode=mode,
            out=dst_filtered
        )

        print(f"  ✓ Time filtering complete\n")

    print(f"{'=' * 70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 70}\n")


def clean_rechunk_time_filtering(embryo_rel_path, dataset_folder, overwrite_original=False):
    """
    Verify that time filtering was successful, delete temporary rechunk files if successful,
    and rechunk the time_filtered zarr back to original chunking for both channels.

    Parameters:
        embryo_rel_path (str): Path relative to dataset_folder for the embryo
        dataset_folder (str): Base dataset folder
        overwrite_original (bool): If True, overwrite the original time_filtered zarr with the rechunked version
    """
    embryo_full_path = os.path.join(dataset_folder, embryo_rel_path, "collated_dataset")

    # Find all original collated_dataset*.zarr files (not rechunk or time_filtered)
    zarr_files = [f for f in glob(os.path.join(embryo_full_path, "collated_dataset*.zarr"))
                  if "_rechunk" not in f and "_time_filtered" not in f]

    if not zarr_files:
        print(f"\n[WARNING] No original collated_dataset*.zarr files found in:\n  {embryo_full_path}\n")
        return

    print(f"\n{'=' * 70}")
    print(f"CLEANING AND RECHUNKING FILTERED DATA")
    print(f"{'=' * 70}")
    print(f"Embryo path: {embryo_rel_path}")
    print(f"Overwrite mode: {'ENABLED' if overwrite_original else 'DISABLED'}")
    print(f"Found {len(zarr_files)} file(s) to process\n")

    for z_file in zarr_files:
        base_name = os.path.basename(z_file)
        rechunk_store = os.path.join(
            embryo_full_path,
            base_name.replace(".zarr", "_rechunk.zarr")
        )
        filtered_store = os.path.join(
            embryo_full_path,
            base_name.replace(".zarr", "_time_filtered.zarr")
        )
        final_store = os.path.join(
            embryo_full_path,
            base_name.replace(".zarr", "_time_filtered_rechunked.zarr")
        )

        print(f"\n{'-' * 70}")
        print(f"Processing: {base_name}")
        print(f"{'-' * 70}")

        # Check if filtered file exists
        if not os.path.exists(filtered_store):
            print(f"  ✗ Time filtered file not found - skipping")
            continue

        # Check if already processed (final rechunked version exists or overwrite already done)
        if not overwrite_original and os.path.exists(final_store):
            print(f"  ✓ Final rechunked file already exists - skipping")
            # Still clean up rechunk file if it exists
            if os.path.exists(rechunk_store):
                print(f"  → Deleting temporary rechunk file...")
                shutil.rmtree(rechunk_store)
                print(f"  ✓ Temporary files removed")
            continue

        # Verify the filtered file is valid
        try:
            original = zarr.open(z_file, mode='r')
            filtered = zarr.open(filtered_store, mode='r')

            # Check shapes match
            if original.shape != filtered.shape:
                print(f"  ✗ Shape mismatch: original {original.shape} vs filtered {filtered.shape}")
                continue

            # Check data is not all zeros or NaN
            sample_data = filtered[0:min(5, filtered.shape[0])]
            if np.all(sample_data == 0) or np.all(np.isnan(sample_data)):
                print(f"  ✗ Invalid filtered data detected (all zeros or NaN)")
                continue

            print(f"  ✓ Verification successful")

            # Delete temporary rechunk file
            if os.path.exists(rechunk_store):
                print(f"  → Deleting temporary rechunk file...")
                shutil.rmtree(rechunk_store)
                print(f"  ✓ Temporary files removed")

            # Rechunk filtered file back to original chunking
            if overwrite_original:
                temp_store = os.path.join(
                    embryo_full_path,
                    base_name.replace(".zarr", "_time_filtered_temp.zarr")
                )

                print(f"  → Rechunking to original chunks (will overwrite)...")
                os.makedirs(os.path.dirname(temp_store), exist_ok=True)

                dst_temp = zarr.open_array(
                    store=temp_store,
                    mode="w",
                    shape=filtered.shape,
                    chunks=original.chunks,
                    dtype=filtered.dtype,
                )
                dst_temp[:] = filtered[:]
                os.sync()

                print(f"  → Replacing original with rechunked version...")
                shutil.rmtree(filtered_store)
                os.rename(temp_store, filtered_store)
                print(f"  ✓ Overwrite complete")
            else:
                if not os.path.exists(final_store):
                    print(f"  → Rechunking to original chunks...")
                    os.makedirs(os.path.dirname(final_store), exist_ok=True)

                    dst_final = zarr.open_array(
                        store=final_store,
                        mode="w",
                        shape=filtered.shape,
                        chunks=original.chunks,
                        dtype=filtered.dtype,
                    )
                    dst_final[:] = filtered[:]
                    os.sync()
                    print(f"  ✓ Rechunking complete")
                else:
                    print(f"  ✓ Final rechunked file already exists")

        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {str(e)}")
            continue

    print(f"\n{'=' * 70}")
    print(f"CLEANING COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    in_store = "/mnt/Data4/Yovan/sols_pipeline/destriped_time_channel_0.ome.zarr"
    out_store = "/mnt/Data4/Yovan/sols_pipeline/median_test_destriped_time_filtered_channel_0.ome.zarr"

    # in_store = "/mnt/Data4/Yovan/sols_pipeline/full_matched_filter_channel_0.ome.zarr"
    # out_store = "/mnt/Data4/Yovan/sols_pipeline/full_matched_filter_time_filter_channel_0.ome.zarr"

    img = zarr.open(in_store)["data"] # type: ignore

    # Preallocate zarr array

    time_chunk = 42

    os.makedirs(os.path.dirname(out_store), exist_ok=True)
    root = zarr.open(out_store, mode="w")
    chunks = (time_chunk, *img.shape[1:]) # type: ignore

    dst = root.create_array( # type: ignore
        "data",
        shape=img.shape, # type: ignore
        chunks=chunks,
        dtype=np.float32, # type: ignore
        overwrite=True,
    )
    
    window = 11

    # sigma = 2
    # x = np.arange(-window//2 + 1, (window//2) + 1)
    # gaussian_kernel = np.exp(-0.5*(x/sigma) ** 2)

    filtered = temporal_filter_zarr(
        img,
        window=11,
        # weights=gaussian_kernel,
        mode="median",
        out=dst,
        chunk_len=21,
    )


    # import dask.array as da
    # import napari
    # check_time_filter = da.from_zarr(filtered)

    # viewer = napari.Viewer()
    # viewer.add_image(check_time_filter)