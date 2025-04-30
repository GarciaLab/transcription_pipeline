import numpy as np

# from numba import jit


def _construct_histogram(trace, weights):
    """
    Constructs weighted histogram for index linked histogram data structure. This is
    as per Weinmann et al. (2014), where the histogram assigns to each trace value
    the sum of the weights of all data points with the same value. Note that we use
    arrays for all datasets instead of linked lists - despite the higher time
    complexity, the tradeoff of being able to use Numpy structures instead of Python
    seems to work out positively.

    In the language of Weinmann et al. (2014), `trace_values` is the sorted list of
    unique values in the input trace, `histogram_values` is the corresponding weight
    of each value in the histogram, and `index_pointer` is an array of indices that
    maps each point of the trace onto its corresponding histogram node.
    """
    # Array splitting code taken from https://stackoverflow.com/a/30003565

    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(trace)

    # sorts records array so all unique elements are together
    sorted_trace = trace[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the
    # count for each element
    trace_values, idx_start, idx_inverse = np.unique(
        sorted_trace, return_index=True, return_inverse=True
    )

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    # find the weights for each unique value
    histogram_weights = [weights[hist_index] for hist_index in res]

    # Compute histogram
    histogram_values = np.array(
        [weights_set.sum() for weights_set in histogram_weights]
    )

    # Link trace points with corresponding histogram nodes
    rev_sort = np.empty_like(trace, dtype=int)
    rev_sort[idx_sort] = np.arange(trace.size)
    index_pointer = idx_inverse[rev_sort].astype(int)

    return trace_values, histogram_values, index_pointer


def _histogram_median(weights):
    """
    Finds the index of the median node of the histogram weights.
    """
    cumsum = np.cumsum(weights)
    median_index = np.argmax(cumsum >= cumsum[-1] / 2)
    return median_index


def l1_potts_step_detection(trace, gamma, weights=None):
    """
    Step detection by minimization of the L1-Potts functional as per algorithm 1 of
    Weinmann et al. (2014). We try to keep the variable names and structure consistent
    with the notation used in the paper.

    :param trace: Time-series of signal for step detection, ignoring any missing values.
        This corresponds to the data vector $(f_1, f_2,..., f_r)$ in the language of
        Weinmann et al. (2014).
    :type trace: np.ndarray
    :param float gamma: Parameter in L1 Potts functional - this controls the balance
        between the term in the L1 Potts loss function that pernalizes the number of
        jumps and the term that penalizes deviation from the data. Should be provided
        as negative - in the limit of negative gamma, the trace perfectly matches the
        data and in the limit of zero gamma, the trace is a flat line through the
        median.
    :param weights: Weight associated with each data fidelity L1 penalty term. If the
        errors in trace values are reliably estimated, weighing by inverse variance
        can help the step calling. Setting `weights=None` assumes data points are
        equally weighed.
    :type weights: np.ndarray
    :return: Function minimizing the L1 Potts functional.
    :rtype: np.ndarray
    """
    # Initialize weights if `None`. Normalization to unity sum is just to ensure
    # consistent order of magnitude of the gamma parameter
    if weights is None:
        weights = np.ones_like(trace) / trace.size
    else:
        weights /= weights.sum()

    n = trace.size
    P = np.zeros(n + 1)
    P[0] = -gamma

    # Initialize index linked histogram. We pad the histogram with nan's to the length
    # of the trace so that insertion operations can be done within the array in
    # contiguous memory.
    H = np.empty_like(np.unique(trace))
    H[:] = np.nan
    H_weights = H.copy()
    I = np.zeros_like(trace, dtype=np.uint)

    Z = np.zeros_like(trace, dtype=int)

    for r in range(n):
        # Init candidate Potts value
        P[r + 1] = np.inf

        f_r = trace[r]
        w_r = weights[r]

        # Insert trace element with corresponding weight to histogram H
        hist_value_mask = H == f_r
        if np.any(hist_value_mask):
            H_weights[hist_value_mask] += w_r
            I[r] = np.argmax(hist_value_mask)
        else:
            insert_position = np.searchsorted(H, f_r)

            H[(insert_position + 1) :] = H[insert_position:-1]
            H[insert_position] = f_r

            H_weights[(insert_position + 1) :] = H_weights[insert_position:-1]
            H_weights[insert_position] = w_r

            I[r] = insert_position

        # Set temporary values of histogram H to permanent ones
        H_temp = H.copy()
        H_weights_temp = H_weights.copy()
        I_temp = I.copy()

        # Index of median node of histogram
        cumsum_weights = np.nancumsum(H_weights)
        M = np.argmax(cumsum_weights > cumsum_weights[-1] / 2)

        # Init median and median deviation
        mu = H_temp[M]  # Value of median node
        d = (weights[:r] * np.abs(trace[:r] - mu)).sum()  # Median deviation

        # Main loop
        for l in range(-1, r):
            # Compute candidate Potts value
            p = P[l + 1] + gamma + d

            # Update Potts value if p is lower than previous best Potts value
            # and update right-most jump
            if p <= P[r + 1]:
                P[r + 1] = p
                Z[r] = l

            w_l = trace[l + 1]
            node_l = I_temp[l + 1]

            # Temporarily remove the weight w_l from the node which the I_l points to.
            H_weights_temp[node_l] -= w_l
            if H_weights_temp[node_l] == 0:
                H_weights_temp[(node_l + 1) :] = H_weights_temp[node_l:-1]
                H_temp[node_l + 1 :] = H_temp[node_l:-1]
                I_temp[I_temp > node_l] -= 1

            # Find new median pointer - this is not done using the iterative shifting
            # procedure in the paper so that it can be vectorized using Numpy
            cumsum_weights = np.nancumsum(H_weights_temp)
            M_temp = np.argmax(cumsum_weights > cumsum_weights[-1] / 2)

            # Update median deviation
            mu = H_temp[M_temp]
            d = (weights[:r] * np.abs(trace[:r] - mu)).sum()

    # Reconstruct L1-Potts minimizer from array of rightmost jumps (described in
    # Lemma 3.2 of reference paper.)
    u_bar = np.zeros_like(trace)

    current_index = n - 1
    while current_index > 0:
        rightmost_jump = Z[current_index]
        jump_trace_values = trace[(rightmost_jump + 1) : (current_index + 1)]
        jump_weights_values = weights[(rightmost_jump + 1) : (current_index + 1)]

        _, jump_histogram_weights, _ = _construct_histogram(
            jump_trace_values, jump_weights_values
        )
        step_median_index = _histogram_median(jump_histogram_weights)

        # Interpolate median if number of points in step is even
        if jump_trace_values.size % 2 == 0:
            step_median_value = jump_trace_values[
                step_median_index : (step_median_index + 2)
            ].mean()
        else:
            step_median_value = jump_trace_values[step_median_index]

        u_bar[(rightmost_jump + 1) : (current_index + 1)] = step_median_value

        current_index = rightmost_jump

    return u_bar
