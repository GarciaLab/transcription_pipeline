import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from scipy import stats
import pandas as pd
from IPython.display import display
import emcee
import os
from warnings import warn
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

from transcription_pipeline.spot_analysis import compile_data
from transcription_pipeline.utils import plottable

from scipy.signal import medfilt
from skimage.restoration import denoise_tv_chambolle


def unpack_functions():
    '''
    Unpacking all the fit functions needed in this class

    OUTPUTS
        make_half_cycle: returns y values of a half-cycle function given
            x values and parameters (basal, t_on, t_dwell, rate, t_interp).
            A half-cycle function is an escalator-shaped function whose arguments
            are shown below.
            ARGUMENTS
                basal: the flat region before rising up,
                t_on: the time when it starts rising,
                t_dwell: the time when reaching the flat region after rising up,
                rate: the slope of the rising region,
                t_interp: a list of (interpolated) x values where the y values will be calculated

        fit_half_cycle: fit a half-cycle function to given data points using MCMC
            ARGUMENTS
                MS2: the y values
                timepoints: the x values
                t_interp: np.linspace(min(timepoints), max(timepoints), 1000)
                std_errs: the standard errors in the y values

        fit_all_traces: apply fit_half_cycle to every fluorescence trace.
            ARGUMENTS
                traces: the output of plottable.generate_trace_plot_list(compiled_dataframe)
                tv_denoised_traces: the list [denoise_tv_chambolle(trace[1], weight=1080, max_num_iter=500)
                                                for trace in traces]

        fit_linear: fit a line to given data points using MCMC
            ARGUMENTS
                Same as the arguments for fit_half_cycle

        bin_particles: adds a column to the compiled_dataframe indicating the bins each particle belong to.
            ARGUMENT
                input_dataframe: the dataframe to be binned.
                bin_num: the number of bins to partition the embryo, default is equal to num_bins
            OUTPUT
                1. A dataframe with a new column indicating the bin index of the particle.
                2. An array of the number of particles in each bin
    '''

    # Version with normalization and regularization
    def make_half_cycle(basal, t_on, t_dwell, rate, t_interp):
        half_cycle = np.zeros_like(t_interp)
        half_cycle[t_interp < t_on] = basal
        half_cycle[(t_interp >= t_on) & (t_interp < t_on + t_dwell)] = basal + rate * (
                t_interp[(t_interp >= t_on) & (t_interp < t_on + t_dwell)] - t_on)
        half_cycle[t_interp >= t_on + t_dwell] = basal + rate * t_dwell
        return half_cycle

    def fit_func(params, MS2, timepoints, t_interp):
        return np.interp(timepoints, t_interp, make_half_cycle(*params, t_interp)) - MS2

    def initial_guess(MS2, timepoints):
        # Initial guess for the parameters
        basal0 = MS2[0]
        t_on0 = timepoints[0]
        t_dwell0 = (2 / 3) * (timepoints[-1] - timepoints[0])
        rate0 = 1
        # print(np.max(mean_dy_dx))
        return [basal0, t_on0, t_dwell0, rate0]

    def fit_half_cycle(MS2, timepoints, t_interp, std_errors, max_nfev=3000):
        # Initial guess
        x0 = initial_guess(MS2, timepoints)

        # Parameter bounds
        lb = [np.min(MS2), 0, 0, 0]  # Ensure t_dwell is non-negative
        ub = [np.max(MS2), np.max(timepoints), np.max(timepoints), 1e7]

        # Scaling factors to normalize parameters
        scale_factors = np.array([np.max(MS2), np.max(timepoints), np.max(timepoints), 100])

        # Scaled bounds
        lb_scaled = np.array(lb) / scale_factors
        ub_scaled = np.array(ub) / scale_factors
        x0_scaled = np.array(x0) / scale_factors

        # Scaled fit function
        def fit_func_scaled(params, MS2, timepoints, t_interp):
            params_unscaled = params * scale_factors
            return fit_func(params_unscaled, MS2, timepoints, t_interp)

        # Negative log-likelihood function
        def negative_log_likelihood(params, MS2, timepoints, t_interp, std_errors, reg=1e-3):
            residuals = fit_func_scaled(params, MS2, timepoints, t_interp) / std_errors
            residuals = np.nan_to_num(residuals, nan=1e6, posinf=1e6, neginf=-1e6)
            regularization = reg * np.sum(params[:] ** 2)
            nll = 0.5 * np.sum(residuals ** 2) + regularization
            return nll

        # Initial parameter estimation using least_squares
        res = least_squares(negative_log_likelihood,
                            x0_scaled, bounds=(lb_scaled, ub_scaled),
                            args=(MS2, timepoints, t_interp, std_errors), max_nfev=max_nfev)

        # Define log-probability function for MCMC
        def log_prob(params, MS2, timepoints, t_interp, std_errors, scale_factors, lb_scaled, ub_scaled):
            if np.any(params < lb_scaled) or np.any(params > ub_scaled):
                return -np.inf
            nll = negative_log_likelihood(params, MS2, timepoints, t_interp, std_errors)
            return -nll  # Convert to log-probability

        # MCMC parameters
        nwalkers = 10
        ndim = len(x0_scaled)
        nsteps = 1000
        initial_pos = res.x + 1e-4 * np.random.randn(nwalkers, ndim)
        # Run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(MS2, timepoints,
                                                                        t_interp, std_errors,
                                                                        scale_factors, lb_scaled, ub_scaled))
        # Run MCMC until the acceptance fraction is at least 0.5
        sampler.run_mcmc(initial_pos, nsteps,
                         progress=False, tune=True)

        # Flatten the chain and discard burn-in steps
        flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)

        # Extract and rescale fit parameters
        basal, t_on, t_dwell, rate = np.median(flat_samples, axis=0) * scale_factors

        # Calculate confidence intervals
        CI = np.percentile(flat_samples, [5, 95], axis=0).T * scale_factors[:, np.newaxis]

        return basal, t_on, t_dwell, rate, CI

    def first_derivative(x, y):
        """
        Compute the first discrete derivative of y with respect to x.
        Parameters:
        x (numpy.ndarray): Independent variable data points.
        y (numpy.ndarray): Dependent variable data points.
        Returns:
        numpy.ndarray: Discrete first derivative of y with respect to x.
        """
        dx = np.diff(x)
        dy = np.diff(y)
        dydx = dy / dx

        # Use central differences for the interior points and forward/backward differences for the endpoints
        dydx_central = np.zeros_like(y)
        dydx_central[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        dydx_central[0] = dydx[0]
        dydx_central[-1] = dydx[-1]

        return dydx_central

    def mean_sign_intervals(function):
        """
        Compute the mean of function over intervals where the function has a constant sign.
        Parameters:
        derivative (numpy.ndarray): Array representing the function.
        Returns:
        numpy.ndarray: Array with mean values of the function over intervals with constant sign.
        """
        # Identify where the sign changes
        sign_changes = np.diff(np.sign(function))
        # Get indices where the sign changes
        change_indices = np.where(sign_changes != 0)[0] + 1

        # Initialize the list to hold mean values
        mean_values = []
        start_index = 0

        for end_index in change_indices:
            # Calculate the mean of the current interval
            interval_mean = np.mean(function[start_index:end_index])
            # Append the mean value to the list
            mean_values.extend([interval_mean] * (end_index - start_index))
            # Update the start index
            start_index = end_index

        # Handle the last interval
        interval_mean = np.mean(function[start_index:])
        mean_values.extend([interval_mean] * (len(function) - start_index))

        return np.array(mean_values), change_indices

    # Function to generate fits for all traces
    def fit_all_traces(traces, tv_denoised_traces):
        """
        Fit half-cycles to all traces in the dataset.
        Parameters:
        traces (list): List of traces to fit.
        tv_denoised_traces (list): List of TV denoised traces.
        Returns:
        list: List of tuples with fit parameters for each trace.
        """
        # Initialize the list to hold fit results
        fit_results = []

        # Create new dataframe to store fit results
        dataframe = pd.DataFrame(columns=['particle', 'fit_results'])

        for i in range(len(traces)):
            # Compute the first derivative of TV denoised with respect to time
            dy_dx = first_derivative(traces[i][0], tv_denoised_traces[i])

            # Compute the mean of the first derivative over intervals with constant sign
            mean_dy_dx, change_indices = mean_sign_intervals(dy_dx)

            # Keep datapoints from before first sign change
            try:
                timepoints = traces[i][0][:change_indices[0]]
                MS2 = traces[i][1][:change_indices[0]]
                MS2_std = traces[i][2][:change_indices[0]]

                # Interpolate the timepoints
                t_interp = np.linspace(min(timepoints), max(timepoints), 1000)
            except:
                print(f"Failed to find derivative sign change for trace {traces[i][3]}")
                fit_results.append([None, None, None, None, None])
                dataframe.loc[i] = [traces[i][3], [None, None, None, None, None]]
                continue

            # Compute the fit values
            try:
                basal, t_on, t_dwell, rate, CI = fit_half_cycle(MS2, timepoints, t_interp, MS2_std)
                fit_result = [timepoints, t_interp, MS2, make_half_cycle(basal, t_on, t_dwell, rate, t_interp),
                              [basal, t_on, t_dwell, rate, CI]]

                fit_results.append(fit_result)
                dataframe.loc[i] = [traces[i][3], fit_result]
            except:
                print(f"Failed to fit trace {traces[i][3]}")
                fit_results.append([timepoints, t_interp, MS2, None, None])
                dataframe.loc[i] = [traces[i][3], [timepoints, t_interp, MS2, None, None]]
                continue

        return fit_results, dataframe


    def fit_average_trace(timepoints, MS2, MS2_std, tv_denoised_traces, bin_index):
        """
        The single-trace version of fit_all_traces.
        """
        # timepoints: the 'frame' column from intensity_by_frame multiplied by time_res_min
        # MS2: the 'average_intensity' column
        # tv_denoised_traces: the 'denoised_average_intensity' column
        # MS2_std: the 'std_erfit_resultsr_intensity' column

        fit_result = []

        # Compute the first derivative of TV denoised with respect to time
        dy_dx = first_derivative(timepoints, tv_denoised_traces)

        # Compute the mean of the first derivative over intervals with constant sign
        mean_dy_dx, change_indices = mean_sign_intervals(dy_dx)

        # Keep datapoints from before first sign change
        try:
            timepoints = timepoints[:change_indices[0]]
            MS2 = MS2[:change_indices[0]]
            MS2_std = MS2_std[:change_indices[0]]

            # Interpolate the timepoints
            t_interp = np.linspace(min(timepoints), max(timepoints), 1000)
        except:
            print(f"Failed to find derivative sign change for average trace {bin_index + 1}")

        # Compute the fit values
        try:
            basal, t_on, t_dwell, rate, CI = fit_half_cycle(MS2, timepoints, t_interp, MS2_std)

            fit_result = [timepoints, t_interp, MS2, make_half_cycle(basal, t_on, t_dwell, rate, t_interp),
                          [basal, t_on, t_dwell, rate, CI]]
        except Exception as e:
            print(f"Failed to fit average trace {bin_index + 1}: {e}")
            fit_result = [timepoints, t_interp, MS2, None, None]

        return fit_result


    def fit_linear(MS2, timepoints, t_interp, std_errors, max_nfev=3000):

        # Scaling factors to normalize parameters
        ceiling = lambda x: -(-x // 1)
        slope = lambda y, x: (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))

        estimated_slope = slope(MS2, timepoints)
        order_of_mag_slope = ceiling(np.log10(estimated_slope))
        slope_scale_factor = 10 ** order_of_mag_slope

        scale_factors = np.array([slope_scale_factor, np.max(timepoints)])

        def initial_guess(MS2, timepoints):
            # initial guess for the parameters
            # dy = np.max(MS2)-np.min(MS2)
            # dx = np.max(timepoints) - np.min(timepoints)
            slope0 = slope(MS2, timepoints)

            intercept0 = np.min(MS2) - np.min(timepoints)

            return [slope0, intercept0]

        # x0 = [(np.max(MS2)-np.min(MS2))/(np.max(timepoints) - np.min(timepoints)), np.min(MS2)]
        x0 = initial_guess(MS2, timepoints)
        x0_scaled = x0 / scale_factors

        f = lambda k, b, x: k * x + b

        # Fit function
        def fit_func(params, MS2, timepoints, t_interp):
            return np.interp(timepoints, t_interp, f(*params, t_interp)) - MS2

        # Scaled fit function
        def fit_func_scaled(params, MS2, timepoints, t_interp):
            params_unscaled = params * scale_factors
            return fit_func(params_unscaled, MS2, timepoints, t_interp)

        # Negative log-likelihood function
        def negative_log_likelihood(params, MS2, timepoints, t_interp, std_errors, reg=1e-3):
            residuals = fit_func_scaled(params, MS2, timepoints, t_interp) / std_errors
            residuals = np.nan_to_num(residuals, nan=1e6, posinf=1e6, neginf=-1e6)
            regularization = reg * np.sum(params[:] ** 2)
            nll = 0.5 * np.sum(residuals ** 2) + regularization
            return nll

        # Initial parameter estimation using least_squares
        res = least_squares(negative_log_likelihood, x0_scaled,
                            args=(MS2, timepoints, t_interp, std_errors), max_nfev=max_nfev)

        # Define log-probability function for MCMC
        def log_prob(params, MS2, timepoints, t_interp, std_errors):
            nll = negative_log_likelihood(params, MS2, timepoints, t_interp, std_errors)
            return -nll  # Convert to log-probability

        # MCMC parameters
        nwalkers = 10
        ndim = len(x0)
        nsteps = 1000
        initial_pos = res.x + 1e-4 * np.random.randn(nwalkers, ndim)
        # Run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(MS2, timepoints,
                                                                        t_interp, std_errors))
        # Run MCMC until the acceptance fraction is at least 0.5
        sampler.run_mcmc(initial_pos, nsteps,
                         progress=False, tune=True)

        # Flatten the chain and discard burn-in steps
        flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)

        # Extract and rescale fit parameters
        slope, intercept = np.median(flat_samples, axis=0) * scale_factors

        # Calculate confidence intervals
        CI = np.percentile(flat_samples, [5, 95], axis=0).T * scale_factors

        return slope, intercept, CI


    def bin_particles(input_dataframe, bin_num):
        '''
        Add a column to the compiled_dataframe for particles indicating the bins they belong to.

        ARGUMENT
            input_dataframe: the dataframe to be binned.
            bin_num: the number of bins to partition the embryo, default is equal to num_bins

        OUTPUT
            1. A dataframe with a new column indicating the bin index of the particle.
            2. An array of the number of particles in each bin
        '''

        dataframe = input_dataframe.copy()

        bin_width = 1 / bin_num

        # Create an array to store the bin indices for each trace
        bin_indices = np.zeros(len(dataframe))

        # Loop through dataframe and assign each trace to a bin based on mean ap position
        for i in range(len(dataframe)):
            particle = dataframe['particle'][i]
            bin_indices[i] = (
                    dataframe.loc[dataframe['particle'] == particle, 'ap']
                    .values[0].mean() // bin_width
            )

        dataframe['bin'] = bin_indices.astype(int)

        # Calculate the number of traces in each bin
        bin_counts = np.zeros(bin_num)
        for i in range(bin_num):
            bin_counts[i] = np.sum(bin_indices == i)

        return dataframe, bin_counts

    return make_half_cycle, fit_half_cycle, fit_all_traces, fit_average_trace, fit_linear, bin_particles

class FitAndAverage:
    '''
    Extract and process the slope values at the rising part of fluorescence
    trace trajectories based on the compiled pandas dataframe generated from
    the spot analysis pipeline and the full embryo pipeline.

    ARGUMENT
        compiled_dataframe: the pandas dataframe generated from
            spot_analysis.compile_data.compile_traces() and
            fullEmbryo_pipeline.FullEmbryo.xy_to_ap().
        nc14_start_frame: the frame where NC14 starts
        min_frames: the minimum frame a trace should have.
            All traces with frame number below this value will be filtered out.
        bin_num: number of bins in total along the AP axis
        dataset_folder_path
    '''

    def __init__(self, compiled_dataframe, nc14_start_frame, min_frames, bin_num, dataset_folder_path):
        self.compiled_dataframe = compiled_dataframe
        self.nc14_start_frame = nc14_start_frame
        self.min_frames = min_frames
        self.bin_num = bin_num
        self.dataset_name = dataset_folder_path

        self.compiled_dataframe_fits = None
        self.compiled_dataframe_fits_checked = None

        self.checked_particle_fits_file_path = self.dataset_name + '/particle_fits_checked.pkl'
        self.checked_particle_fits_previous = os.path.isfile(self.checked_particle_fits_file_path)

        (make_half_cycle, fit_half_cycle, fit_all_traces,
         fit_average_trace, fit_linear, bin_particles) = unpack_functions()

        if self.checked_particle_fits_previous:
            # Load the DataFrame from the .pkl file
            print('Load previous particle trace fit checking results from "particle_fits_checked.pkl"')
            traces_compiled_dataframe_fits_checked = pd.read_pickle(self.checked_particle_fits_file_path)
            traces_compiled_dataframe_fits_checked_temp = traces_compiled_dataframe_fits_checked.copy()

            self.compiled_dataframe_fits = traces_compiled_dataframe_fits_checked_temp
            self.compiled_dataframe_fits_checked = traces_compiled_dataframe_fits_checked

        else:
            print('No previous particle trace fit checking results detected. Do particle trace fitting for the dataframe.')

            # Fit to the current dataframe instead of loading from previous result
            nc14_start_frame = self.nc14_start_frame
            compiled_dataframe = self.compiled_dataframe
            min_frames = self.min_frames
            # Restrict to longer traces
            traces_compiled_dataframe = compiled_dataframe[
                compiled_dataframe["frame"].apply(lambda x: x.size) > min_frames
                ]
            # Restrict to traces starting at frame nc14_start_frame and above
            traces_compiled_dataframe = traces_compiled_dataframe[
                traces_compiled_dataframe["frame"].apply(lambda x: x[0] >= nc14_start_frame)
            ]

            # Order the traces based on the mean x position
            traces_compiled_dataframe = traces_compiled_dataframe.sort_values(
                by="x", key=lambda x: x.apply(np.mean)
            )

            traces = plottable.generate_trace_plot_list(traces_compiled_dataframe)

            # Generate TV denoised traces
            tv_denoised_traces = [
                denoise_tv_chambolle(trace[1], weight=1080, max_num_iter=500) for trace in traces
            ]

            # Generate fits for all traces
            fit_results, dataframe = fit_all_traces(traces, tv_denoised_traces)

            print(f"Number of traces: {len(traces)}")

            # Show number of traces with valid fits
            print(f"Number of traces with valid fits: {sum([result[4] is not None for result in fit_results])}")

            # Show number of traces with invalid fits
            print(f"Number of traces with invalid fits: {sum([result[4] is None for result in fit_results])}")

            traces_compiled_dataframe_fits = pd.merge(traces_compiled_dataframe, dataframe, on='particle', how='inner')

            # Add columns: the approval status, denoised trace, and fitted rate of the particle
            length = traces_compiled_dataframe_fits.index.max()
            status = [0 for _ in range(length + 1)]
            modified_fit_results = [None for _ in range(length + 1)]
            traces_compiled_dataframe_fits['modified_fit_results'] = modified_fit_results
            traces_compiled_dataframe_fits['tv_denoised_trace'] = tv_denoised_traces
            traces_compiled_dataframe_fits['approval_status'] = status

            self.compiled_dataframe_fits = traces_compiled_dataframe_fits

    def check_particle_fits(self, show_denoised_plot=False):
        '''
        Check the fit of each particle.

        ARGUMENT
            binned_particles_fitted: a pandas dataframe containing the bin average fit data.
            show_denoised_plot: show the denoised trace of the data points, default is False.

        KEY INPUTS
            left (<-): move to previous bin
            right (->): move to next bin
            a: accept fit for this bin (plot turns green)
            r: reject fit for this bin (plot turns red)
            c: clear approval status (plot returns white)
            i: move to the particle with the given particle index
            j: jump to the n-th particle in the list (with n given)
            f: redo a linear fit based on data within a certain x range of input
               (plot turns yellow)
        '''

        (make_half_cycle, fit_half_cycle, fit_all_traces,
         fit_average_trace, fit_linear, bin_particles) = unpack_functions()

        binned_particles_fitted = self.compiled_dataframe_fits

        fig, ax = plt.subplots()

        particle_index = 0
        particle_num = binned_particles_fitted.index.max()
        x = None
        y = None
        y_denoised = None
        y_err = None

        # move to the first unchecked particle--------------------------------------
        first_flag = False
        while not first_flag:
            particle_data = binned_particles_fitted[particle_index:particle_index + 1]
            status = particle_data['approval_status'].values[0]
            if status == 0:
                first_flag = True
                if not particle_index == 0:
                    print(f'Moved to particle {particle_index + 1} out of {particle_num}, the first unchecked particle')
            else:
                if particle_index < particle_num-1:
                    particle_index += 1
                elif particle_index == particle_num-1:
                    warn('No particle has been left unchecked')
                    particle_index = 0
                    break

        # ---------------------------------------------------------------------------

        def update_plot(particle_index):
            nonlocal x, y, y_denoised, y_err
            ax.clear()

            particle_data = binned_particles_fitted[particle_index:particle_index + 1]  # select the particle
            try:
                x = particle_data['t_s'].values[0]
                y = particle_data['intensity_from_neighborhood'].values[0]
                y_denoised = particle_data['tv_denoised_trace'].values[0]
                y_err = particle_data['intensity_std_error_from_neighborhood'].values[0]

                # plot the particle trace with error bar along with the denoised trace
                ax.errorbar(x / 60, y, yerr=y_err, fmt=".", elinewidth=1, label='Data')
                if show_denoised_plot:
                    ax.plot(x, y_denoised, color='k', label='TV denoised')

                # plot the fit

                try:
                    # plot the modified linear fit
                    modified = (particle_data['approval_status'].values == 2)
                    if modified:
                        fit_result = particle_data['modified_fit_results'].values[0]
                        timepoints, t_interp, MS2, fit, [intercept, _, _, rate, CI] = fit_result
                        ax.errorbar(timepoints / 60, MS2, fmt=".", elinewidth=1, label='Selected data for modified fit')

                    # plot the original half cycle fit
                    else:
                        fit_result = particle_data['fit_results'].values[0]
                        timepoints, t_interp, MS2, fit, [basal, t_on, t_dwell, rate, CI] = fit_result

                    if modified:
                        ax.plot(t_interp / 60, fit, label=f'Modified fit (slope = {round(rate * 60, 2)} AU/min)',
                                linewidth=3)
                    else:
                        ax.plot(t_interp / 60, fit, label=f'Fit (slope = {round(rate * 60, 2)} AU/min)', linewidth=3)

                    # ax.plot(t_interp, make_half_cycle(basal, t_on, t_dwell, rate, t_interp), label=f"Fit (slope = {round(rate, 2)})", linewidth=3, color='orange')
                except:
                    pass

                particle = particle_data['particle'].values[0]
                # bin = particle_data['bin'].values[0]
                mean_x = (particle_data.loc[particle_data["particle"] == particle, "x"]
                          .values[0]
                          .mean()
                          )
                initial_frame = (binned_particles_fitted.loc[binned_particles_fitted["particle"] == particle, "frame"].
                values[0][0]
                )
                status = particle_data['approval_status'].values[0]

                if status == 1:
                    ax.set_facecolor((0.7, 1, 0.7))  # approve
                elif status == -1:
                    ax.set_facecolor((1, 0.7, 0.7))  # reject color
                elif status == 0:
                    ax.set_facecolor((1, 1, 1))
                elif status == 2:
                    ax.set_facecolor((1, 1, 0.7))

                ax.set_title(
                    f'Particle #{particle} ({particle_index + 1}/{particle_num + 1}), x = {np.round(mean_x, 2)}, Initial frame {initial_frame}')
                ax.set_xlabel("Time (min)")
                ax.set_ylabel("Spot intensity (AU)")
                ax.legend()

            except Exception as e:
                particle = particle_data['particle'].values[0]
                print(f"Error processing particle {particle}: {e}")

            fig.canvas.draw()

        def on_key(event):
            nonlocal particle_index
            nonlocal x, y, y_denoised, y_err

            if event.key == 'left':
                # move to the previous particle
                particle_index = max(0, particle_index - 1)
            elif event.key == 'right':
                # move to the next particle
                particle_index = min(len(binned_particles_fitted) - 1, particle_index + 1)
            elif event.key == 'a':
                # accept the fit for this particle
                binned_particles_fitted.at[particle_index, 'approval_status'] = 1
            elif event.key == 'r':
                # reject the fit for this particle
                binned_particles_fitted.at[particle_index, 'approval_status'] = -1
            elif event.key == 'c':
                # clear the approval status for this particle
                binned_particles_fitted.at[particle_index, 'approval_status'] = 0
                binned_particles_fitted.at[particle_index, 'modified_fit_results'] = None
            elif event.key == 'i':
                # Move to the particle with the particle index given
                # Create a Tkinter root window and hide it
                root = tk.Tk()
                root.withdraw()

                index_array = np.sort(binned_particles_fitted['particle'].values)
                # Ask for input
                input_index = simpledialog.askinteger("Input", f"Choose particle index from {index_array}:",
                                                      minvalue=0)  # , maxvalue=particle_num)

                # Update the particle index if input is valid
                if input_index is not None:
                    particle_index = \
                    binned_particles_fitted[binned_particles_fitted['particle'] == input_index].index.values[0]

                # Destroy the Tkinter root window
                root.destroy()

            elif event.key == 'j':
                # Jump to the nth particle in the list
                # Create a Tkinter root window and hide it
                root = tk.Tk()
                root.withdraw()

                # Ask for input
                input_index = simpledialog.askinteger("Input", f"Jump to particle __ out of {particle_num + 1}:",
                                                      minvalue=1, maxvalue=particle_num + 1)

                # Update the particle index if input is valid
                if input_index is not None:
                    particle_index = input_index - 1

                # Destroy the Tkinter root window
                root.destroy()

            elif event.key == 'f':
                # Fit based on a chosen range
                def get_two_numbers():
                    # Create a window
                    root = tk.Tk()
                    root.withdraw()  # Hide the root window

                    # Ask for the first number
                    num1 = simpledialog.askfloat("Input the range to fit a line", "Enter the left bound of the range:")

                    # Ask for the second number
                    num2 = simpledialog.askfloat("Input the range to fit a line", "Enter the right bound of the range:")

                    return num1, num2

                x_min, x_max = get_two_numbers()

                # Select points within the range of x_min and x_max
                mask = (x >= x_min * 60) & (x <= x_max * 60)
                new_x = x[mask]
                new_y = y[mask]
                new_yerr = y_err[mask]

                x_interp = np.linspace(min(new_x), max(new_x), 1000)

                # The linear regime------------------------------------------
                f = lambda k, b, x: k * x + b
                try:

                    slope, intercept, CI = fit_linear(new_y, new_x, x_interp, new_yerr)

                    slope_conf_interval = CI[0]
                    intercept_conf_interval = CI[1]

                    particle_fit_result_modified = [new_x, x_interp, new_y, f(slope, intercept, x_interp),
                                                    [intercept, np.nan, np.nan, slope,
                                                     np.array([intercept_conf_interval, [np.nan, np.nan],
                                                               [np.nan, np.nan], slope_conf_interval])]]

                    binned_particles_fitted.at[particle_index, 'modified_fit_results'] = particle_fit_result_modified

                    binned_particles_fitted.at[particle_index, 'approval_status'] = 2

                except:
                    pass

            update_plot(particle_index)

        update_plot(particle_index)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    def save_checked_particle_fits(self):
        '''Save the checked particle fits'''

        if self.checked_particle_fits_previous:
            traces_compiled_dataframe_fits_checked_temp = self.compiled_dataframe_fits
            traces_compiled_dataframe_fits_checked = self.compiled_dataframe_fits_checked
            # Check if any changes is made to the approval status of the particles
            if all(traces_compiled_dataframe_fits_checked_temp.approval_status == traces_compiled_dataframe_fits_checked.approval_status):
                print('No changes made to the particle fit checking results')
            else:
                answer = messagebox.askyesno('Question', 'Changes to the checked particle fits detected. Save the changes?')
                if answer:
                    traces_compiled_dataframe_fits_checked = traces_compiled_dataframe_fits_checked_temp
                    traces_compiled_dataframe_fits_checked.to_pickle(self.checked_particle_fits_file_path, compression=None)
                    print('Checked particle fits updated')
                else:
                    print('No changes made to the particle fit checking results')

        else:
            traces_compiled_dataframe_fits = self.compiled_dataframe_fits
            traces_compiled_dataframe_fits_checked = traces_compiled_dataframe_fits.reset_index()
            traces_compiled_dataframe_fits_checked.to_pickle(self.checked_particle_fits_file_path, compression=None)
            print('Checked particle fits saved')

        self.compiled_dataframe_fits_checked = traces_compiled_dataframe_fits_checked

    def average_particle_fits(self, plot=True):
        '''
        Take the average of all the approved fits of individual particles in each bin,
        with an option to plot them by bins.

        OUTPUTS
            1. A dictionary giving the following details:
                - The bin index
                - The number of particles in that bin
                - The particle IDs in the bin
                - The fit rate of each particle in the bin
            2. An array of average fit rates for each bin
            3. An array of the number of particles in each bin
        '''
        # bin_particles: a function that adds a column to the dataframe indicating the bins the particles belong to. Default bin number is equal to num_bins

        (make_half_cycle, fit_half_cycle, fit_all_traces,
         fit_average_trace, fit_linear, bin_particles) = unpack_functions()

        traces_compiled_dataframe_fits_checked = self.compiled_dataframe_fits_checked


        # compute_average_fit_rates_for_bins: a function that calculates the average fit rates for each bin
        def compute_average_fit_rates_for_bins(input_dataframe, bin_num, plot_result=plot):
            '''
            Calculate the average fit rates for each bin and store particle
            IDs with rates in each bin.

            ARGUMENT
                input_dataframe: a particle dataframe
                bin_num: the number of bins to partition the embryo, default is equal to num_bins

            OUTPUTS
                1. A dictionary giving the following details:
                    - The bin index
                    - The number of particles in that bin
                    - The particle IDs in the bin
                    - The fit rate of each particle in the bin
                2. An array of average fit rates for each bin
                3. An array of the number of particles in each bin
            '''

            # Bin the particles
            approved_mask = (input_dataframe['approval_status'] == 1) | (input_dataframe['approval_status'] == 2)

            approved_dataframe = input_dataframe[approved_mask].reset_index()

            approved_binned_dataframe, bin_counts = bin_particles(approved_dataframe, bin_num)

            bin_indices = approved_binned_dataframe['bin'].values

            # Sorting traces by the bin they belong to, and calculate the average of fit slope for each bin
            mean_fit_rates = np.zeros(bin_num)
            bin_particles_rates = np.zeros(bin_num, dtype=object)
            SE_fit_rates = np.zeros(bin_num)

            for i in range(bin_num):
                # pass bins that has no particles
                if bin_counts[i] == 0:
                    mean_fit_rates[i] = np.nan
                    continue
                else:
                    bin_i_dataframe = approved_binned_dataframe[bin_indices == i]

                    rates = (
                            60 * bin_i_dataframe['fit_results'].apply(
                        lambda x: x[4][3] if x[4] is not None else np.nan).values
                    )

                    # Replace the old fits with the modified linear fits
                    modified_mask = (bin_i_dataframe['approval_status'] == 2)

                    modified_rates = 60 * bin_i_dataframe.loc[modified_mask, 'modified_fit_results'].apply(
                        lambda x: x[4][3] if x[4] is not None else np.nan).values

                    rates[modified_mask] = modified_rates

                    particles = (
                        approved_binned_dataframe.loc[bin_indices == i, 'particle']
                        .values
                    )

                    # Store the particle IDs with their rates in each bin for further analysis
                    bin_particles_rates[i] = {
                        'bin': i+1,
                        'bin_ap_position': i/self.bin_num,
                        'bin_particle_counts': bin_counts[i],
                        'particles': particles,
                        'rates': rates,
                        'mean_rate': np.nanmean(rates),
                        'SE_rate': np.nanstd(rates) / np.sqrt(len(rates))
                    }

                    mean_fit_rates[i] = (np.nanmean(rates))

                    # Standard error of the mean
                    SE_fit_rates[i] = np.nanstd(rates) / np.sqrt(len(rates))

            # Prepare the data for plotting
            not_nan = ~np.isnan(mean_fit_rates)

            bin_indices = np.arange(bin_num)
            ap_positions = bin_indices * 1 / bin_num

            bin_slopes = mean_fit_rates[not_nan]
            bin_slope_errs = SE_fit_rates[not_nan]

            max_bin_slope = np.max(bin_slopes)
            ylim_up = 1.5 * max_bin_slope

            if plot_result:
                # Plot the average slope of trace fits for each bin number
                plt.figure()
                plt.errorbar(ap_positions[not_nan], bin_slopes, yerr=bin_slope_errs, capsize=2, fmt='o')
                plt.xlabel('AP Position')
                plt.ylabel('Average rate of trace fits (AU/min)')
                plt.title('Average rate of trace fits vs. AP position')
                plt.ylim(0, ylim_up)
                plt.show()

            return ap_positions, mean_fit_rates, SE_fit_rates, bin_counts, bin_particles_rates

        ap_positions, mean_fit_rates, SE_fit_rates, bin_counts, bin_particles_rates = (
            compute_average_fit_rates_for_bins(traces_compiled_dataframe_fits_checked, self.bin_num))

        return ap_positions, mean_fit_rates, SE_fit_rates, bin_counts, bin_particles_rates


class AverageAndFit:
    '''
    time_bin_width: please use dataset.export_frame_metadata[0]['t_s'][1, 0]
    '''
    def __init__(self, compiled_dataframe, nc14_start_frame, time_bin_width, bin_num, dataset_folder_path):
        self.compiled_dataframe = compiled_dataframe
        self.nc14_start_frame = nc14_start_frame
        self.time_bin_width = time_bin_width
        self.bin_num = bin_num
        self.dataset_name = dataset_folder_path

        self.checked_bin_fits_file_path = self.dataset_name + '/bin_fits_checked.pkl'
        self.checked_bin_fits_previous = os.path.isfile(self.checked_bin_fits_file_path)

        (make_half_cycle, fit_half_cycle, fit_all_traces,
         fit_average_trace, fit_linear, bin_particles) = unpack_functions()

        if self.checked_bin_fits_previous:
            # Load the DataFrame from the .pkl file
            print('Load previous bin fit checking results from "bin_fits_checked.pkl"')
            bin_fits_checked = pd.read_pickle(self.checked_bin_fits_file_path)
            bin_fits_checked_temp = bin_fits_checked.copy()

            self.bin_average_fit_dataframe = bin_fits_checked_temp
            self.bin_average_fit_dataframe_checked = bin_fits_checked

        else:
            print('No previous bin fit checking results detected. Do bin fitting for the dataframe.')

            dataframe_nc14 = self.compiled_dataframe[self.compiled_dataframe['frame'].apply(lambda x: x[0] >= nc14_start_frame)].reset_index()

            binned_dataframe_nc14, _ = bin_particles(dataframe_nc14, self.bin_num)

            binned_particles_nc14 = [None] * self.bin_num
            for bin in range(self.bin_num):
                binned_particles_nc14[bin] = binned_dataframe_nc14[binned_dataframe_nc14['bin'] == bin]

            bin_particle_num = [0] * self.bin_num
            for bin in range(self.bin_num):
                bin_particle_num[bin] = len(binned_particles_nc14[bin])

            # bin_average_over_time_bins: a function taking the average of MS2 signals for each time bin along the time axis for an AP bin

            def bin_average_over_time_bins(bin_dataframe, time_bin_width=self.time_bin_width,
                                           shift_traces_to_same_start_time=True):

                '''
                This function should be applied on a dataframe of particles for a particular bin.
                What this function does:
                1. Split the time axis into time bins
                2. Average all MS2 signals in each time bin

                ARGUMENT
                    bin_dataframe: a pandas dataframe for all the particles in an AP bin
                    time_bin_width: float, the width of the time bin, default is the frame duration of the dataset
                    shift_traces_to_same_start_time: bool, if true, shift the time array for each particle to start at 0.

                OUTPUT
                    1. bin_centers: an array of the time bin centers
                    2. bin_means: an array of the mean MS2 signal in each time bin
                    3. bin_stds: an array of the standard deviation for the MS2 signals in each time bin
                '''

                # extract the time column
                t_s = bin_dataframe['t_s'].values
                if shift_traces_to_same_start_time:
                    t_s_shifted = [None] * len(t_s)
                    for i in range(len(t_s)):
                        t_first = t_s[i][0]
                        shifted_t_s = t_s[i] - t_first
                        t_s_shifted[i] = shifted_t_s
                    t_s = t_s_shifted

                # extract the intensity column
                intensity = bin_dataframe['intensity_from_neighborhood'].values
                # intensity_err = bin_dataframe['intensity_std_error_from_neighborhood'].values

                # Take averages of all the intensity values over time bins:
                # 1. Flatten t_s and intensity into single arrays
                t_s_flat = np.concatenate(t_s)
                intensity_flat = np.concatenate(intensity)

                # 2. Define time bin edges with specified width
                x_min, x_max = min(t_s_flat), max(t_s_flat)
                time_bins = np.arange(x_min, x_max + time_bin_width, time_bin_width)

                # 3. Digitize t_s_flat into time bins
                time_bin_indices = np.digitize(t_s_flat, time_bins)

                # 4. Compute the average and standard deviation for the intensity values in each time bin
                time_bin_means = np.array([intensity_flat[time_bin_indices == i].mean() for i in range(1, len(time_bins))])

                time_bin_stddevs = np.array([intensity_flat[time_bin_indices == i].std() for i in range(1, len(time_bins))])

                time_bin_stderrs = np.array([intensity_flat[time_bin_indices == i].std()
                                             / np.sqrt(len(intensity_flat[time_bin_indices == i]))
                                             for i in range(1, len(time_bins))])

                # 5. Get the time bin centers for plotting
                time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

                # 6. Drop the entries with nan means
                nan_indices = np.isnan(time_bin_means)

                time_bin_centers = time_bin_centers[~nan_indices] / 60
                time_bin_means = time_bin_means[~nan_indices]
                time_bin_stddevs = time_bin_stddevs[~nan_indices]
                time_bin_stderrs = time_bin_stderrs[~nan_indices]

                return np.array([time_bin_centers, time_bin_means, time_bin_stddevs, time_bin_stderrs])


            # Store the bin averages for all AP bins in a pandas dataframe
            self.bin_average_fit_dataframe = pd.DataFrame({'time_bin_centers': [None] * self.bin_num,
                                                      'time_bin_means': [None] * self.bin_num,
                                                      'time_bin_stddevs': [None] * self.bin_num,
                                                      'time_bin_stderrs': [None] * self.bin_num,
                                                      'denoised_time_bin_means': [None] * self.bin_num,
                                                      'bin_fit_result': [None] * self.bin_num,
                                                      'bin_fit_slope': [np.nan for i in range(self.bin_num)],
                                                      'bin_fit_result_modified': [None] * self.bin_num,
                                                      'bin_fit_slope_modified': [np.nan for i in range(self.bin_num)],
                                                      'bin_particle_number': [None] * self.bin_num,
                                                      'approval_status': [0] * self.bin_num})

            for bin in range(self.bin_num):
                self.bin_average_fit_dataframe['bin_particle_number'][bin] = bin_particle_num[bin]
                try:
                    [time_bin_centers, time_bin_means, time_bin_stddevs, time_bin_stderrs] = bin_average_over_time_bins(
                        binned_particles_nc14[bin])
                    self.bin_average_fit_dataframe['time_bin_centers'][bin] = time_bin_centers
                    self.bin_average_fit_dataframe['time_bin_means'][bin] = time_bin_means
                    self.bin_average_fit_dataframe['time_bin_stddevs'][bin] = time_bin_stddevs
                    self.bin_average_fit_dataframe['time_bin_stderrs'][bin] = time_bin_stderrs

                    denoised_time_bin_means = denoise_tv_chambolle(time_bin_means, weight=1080, max_num_iter=500)
                    self.bin_average_fit_dataframe['denoised_time_bin_means'][bin] = denoised_time_bin_means

                    try:
                        bin_fit_result = fit_average_trace(time_bin_centers, time_bin_means, time_bin_stderrs,
                                                           denoised_time_bin_means, bin)
                        self.bin_average_fit_dataframe['bin_fit_result'][bin] = bin_fit_result
                        self.bin_average_fit_dataframe['bin_fit_slope'][bin] = bin_fit_result[4][3]
                    except:
                        self.bin_average_fit_dataframe['approval_status'][bin] = -1
                except:
                    self.bin_average_fit_dataframe['approval_status'][bin] = -1


    def check_bin_fits(self, show_denoised_plot=False, show_fit=True, show_std_dev=True, show_std_err=True):
        '''
        Check the fit for each bin average.

        Note that the bin index starts from 1, while the pandas dataframe starts from 0.

        ARGUMENT
            bin_average_fit_dataframe: a pandas dataframe containing the bin average fit data.
            show_denoised_plot: show the denoised trace of the data points, default is False.
            show_fit: show the half cycle fit for the data points, default is True.
            show_std_dev: show the standard deviation for each data point.
            show_std_err: show the standard error for each data point.

        KEY INPUTS
            left (<-): move to previous bin
            right (->): move to next bin
            a: accept fit for this bin (plot turns green)
            r: reject fit for this bin (plot turns red)
            c: clear approval status (plot returns white)
            f: redo the fit based on data within a certain x range,
               which is determined by clicking twice on the plot
               (plot turns yellow)
        '''

        (make_half_cycle, fit_half_cycle, fit_all_traces,
         fit_average_trace, fit_linear, bin_particles) = unpack_functions()

        bin_average_fit_dataframe = self.bin_average_fit_dataframe

        fig, ax = plt.subplots()
        bin_index = 0
        bin_num = len(bin_average_fit_dataframe)
        x = None
        y = None
        y_stddev = None
        y_stderr = None
        y_denoised = None

        first_nonempty_bin = 0
        while bin_average_fit_dataframe.loc[first_nonempty_bin, 'time_bin_means'] is None:
            first_nonempty_bin += 1
            if first_nonempty_bin == bin_num:
                break

        # move to the first unchecked bin--------------------------------------
        first_flag = False
        while not first_flag:
            bin_data = bin_average_fit_dataframe[bin_index:bin_index + 1]
            status = bin_data['approval_status'].values[0]
            if status == 0:
                first_flag = True
                if not bin_index == 0:
                    print(f'Moved to bin {bin_index + 1} out of {bin_num}, the first unchecked bin')
            else:
                if bin_index < bin_num - 1:
                    bin_index += 1
                elif bin_index == bin_num - 1:
                    warn('No bin has been left unchecked. Move to the first bin with data')
                    bin_index = first_nonempty_bin
                    break

        # ---------------------------------------------------------------------------

        def update_plot(bin_index):
            nonlocal x, y, y_stddev, y_stderr, y_denoised
            ax.clear()
            try:
                x = bin_average_fit_dataframe.loc[bin_index, 'time_bin_centers']
                y = bin_average_fit_dataframe.loc[bin_index, 'time_bin_means']
                y_stddev = bin_average_fit_dataframe.loc[bin_index, 'time_bin_stddevs']
                y_stderr = bin_average_fit_dataframe.loc[bin_index, 'time_bin_stderrs']
                y_denoised = bin_average_fit_dataframe.loc[bin_index, 'denoised_time_bin_means']

                # plot the bin averaged trace with error bar along with the denoised trace
                if show_std_dev:
                    ax.errorbar(x, y, yerr=y_stddev, fmt=".", elinewidth=1, label='Data with std dev', alpha=0.35)
                if show_std_err:
                    ax.errorbar(x, y, yerr=y_stderr, fmt=".", elinewidth=1, label='Data with std err')

                if show_denoised_plot:
                    ax.plot(x, y_denoised, color='k', label='TV denoised')

                # plot the half cycle fit
                try:
                    if bin_average_fit_dataframe.at[bin_index, 'approval_status'] == 2:
                        fit_result = bin_average_fit_dataframe.loc[bin_index, 'bin_fit_result_modified']
                        timepoints, t_interp, MS2, linear_fit, [intercept, _, _, rate, CI] = fit_result
                        ax.errorbar(timepoints, MS2, fmt=".", elinewidth=1, label='Selected data for new fit')

                        if show_fit:
                            ax.plot(t_interp, linear_fit,
                                    label=f'Modified fit (slope = {round(rate, 2)} AU/min)', linewidth=3)
                    else:
                        fit_result = bin_average_fit_dataframe.loc[bin_index, 'bin_fit_result']
                        timepoints, t_interp, MS2, half_cycle_fit, [basal, t_on, t_dwell, rate, CI] = fit_result

                        if show_fit:
                            ax.plot(t_interp, half_cycle_fit, label=f'Fit (slope = {round(rate, 2)} AU/min)',
                                    linewidth=3)

                except:
                    pass

                status = bin_average_fit_dataframe.loc[bin_index, 'approval_status']

                if status == 1:
                    ax.set_facecolor((0.7, 1, 0.7))  # green
                elif status == -1:
                    ax.set_facecolor((1, 0.7, 0.7))  # red
                elif status == 0:
                    ax.set_facecolor((1, 1, 1))
                elif status == 2:
                    ax.set_facecolor((1, 1, 0.7))  # yellow

                ax.set_title(f'Bin {bin_index + 1}/{bin_num}')
                ax.set_xlabel("Time (min)")
                ax.set_ylabel("Spot intensity (AU)")
                ax.legend()

            except Exception as e:
                print(f"Error processing bin {bin_index}: {e}")

            fig.canvas.draw()

        def on_key(event):
            nonlocal bin_index, x, y, y_stddev, y_stderr, y_denoised
            if event.key == 'left':
                bin_index = max(0, bin_index - 1)
            elif event.key == 'right':
                bin_index = min(len(bin_average_fit_dataframe) - 1, bin_index + 1)
            elif event.key == 'a':
                # a for accept/approve
                bin_average_fit_dataframe.at[bin_index, 'approval_status'] = 1
            elif event.key == 'r':
                # r for reject
                bin_average_fit_dataframe.at[bin_index, 'approval_status'] = -1
            elif event.key == 'c':
                # c for clear
                bin_average_fit_dataframe.at[bin_index, 'approval_status'] = 0
                bin_average_fit_dataframe.at[bin_index, 'bin_fit_result_modified'] = None
                bin_average_fit_dataframe.at[bin_index, 'bin_fit_slope_modified'] = np.nan
            elif event.key == 'f':
                # Fit based on a chosen range
                def get_two_numbers():
                    # Create a window
                    root = tk.Tk()
                    root.withdraw()  # Hide the root window

                    # Ask for the first number
                    num1 = simpledialog.askfloat("Input the range to fit a line", "Enter the left bound of the range:")

                    # Ask for the second number
                    num2 = simpledialog.askfloat("Input the range to fit a line", "Enter the right bound of the range:")

                    return num1, num2

                x_min, x_max = get_two_numbers()

                # Select points within the range of x_min and x_max
                mask = (x >= x_min) & (x <= x_max)
                new_x = x[mask]
                new_y = y[mask]
                new_y_stddev = y_stddev[mask]
                new_y_stderr = y_stderr[mask]

                x_interp = np.linspace(min(new_x), max(new_x), 1000)

                # The linear regime------------------------------------------
                f = lambda k, b, x: k * x + b
                try:

                    slope, intercept, CI = fit_linear(new_y, new_x, x_interp, new_y_stderr)

                    slope_conf_interval = CI[0]
                    intercept_conf_interval = CI[1]

                    bin_fit_result_modified = [new_x, x_interp, new_y, f(slope, intercept, x_interp),
                                               [intercept, np.nan, np.nan, slope,
                                                np.array([intercept_conf_interval, [np.nan, np.nan],
                                                          [np.nan, np.nan], slope_conf_interval])]]

                    bin_average_fit_dataframe.at[bin_index, 'bin_fit_result_modified'] = bin_fit_result_modified

                    bin_average_fit_dataframe.at[bin_index, 'bin_fit_slope_modified'] = slope

                    bin_average_fit_dataframe.at[bin_index, 'approval_status'] = 2

                except:
                    pass

            update_plot(bin_index)

        update_plot(bin_index)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()


    def save_checked_bin_fits(self):
        '''Save the checked bin fits'''

        if self.checked_bin_fits_previous:
            bin_fits_checked_temp = self.bin_average_fit_dataframe
            bin_fits_checked = self.bin_average_fit_dataframe_checked
            # Check if any changes is made to the approval status of the particles
            if all(bin_fits_checked_temp.approval_status == bin_fits_checked.approval_status):
                print('No changes made to the bin fit checking results')
            else:
                answer = messagebox.askyesno('Question', 'Changes to the checked bin fits detected. Save the changes?')
                if answer:
                    bin_fits_checked = bin_fits_checked_temp
                    bin_fits_checked.to_pickle(self.checked_bin_fits_file_path, compression=None)
                    print('Checked bin fits updated')
                else:
                    print('No changes made to the bin fit checking results')

        else:
            bin_fits = self.bin_average_fit_dataframe
            bin_fits_checked = bin_fits.reset_index()
            bin_fits_checked.to_pickle(self.checked_bin_fits_file_path, compression=None)
            print('Checked bin fits saved')

        self.bin_average_fit_dataframe_checked = bin_fits_checked


    def plot_bin_fits(self):
        '''
        Plot bin slopes vs AP position

        OUTPUTS
            ap_positions: the AP positions of the bins
            bin_slopes: the slopes of the bins
            bin_slope_errs: the errors in the bin slopes (95% confidence level)
        '''

        bin_indices = np.arange(self.bin_num)

        ap_positions = bin_indices * 1 / self.bin_num

        plot_mask = ((self.bin_average_fit_dataframe['approval_status'] == 1) |
                     (self.bin_average_fit_dataframe['approval_status'] == 2))

        modified_mask = (self.bin_average_fit_dataframe['approval_status'] == 2)

        dataframe_for_plot = self.bin_average_fit_dataframe.copy()

        # Extract slope
        bin_slopes = dataframe_for_plot['bin_fit_slope'].values
        modified_slopes = dataframe_for_plot.loc[modified_mask, 'bin_fit_slope_modified'].values
        bin_slopes[modified_mask] = modified_slopes # replace with modified slope

        # Extract slope stderr
        bin_fit_results = dataframe_for_plot['bin_fit_result'].values
        modified_results = dataframe_for_plot.loc[modified_mask, 'bin_fit_result_modified'].values
        bin_fit_results[modified_mask] = modified_results

        bin_slope_errs = np.zeros((self.bin_num, 2))
        for bin in range(self.bin_num):
            try:
                bin_slope_err = bin_fit_results[bin][-1][-1][-1]
                bin_slope_errs[bin] = bin_slope_err
            except:
                pass

        bin_slope_errs = np.transpose(np.abs(bin_slope_errs - bin_slopes[:, np.newaxis]))


        # Prepare plotting variables
        #bin_indices_for_plot = bin_indices[plot_mask]
        ap_positions_for_plot = ap_positions[plot_mask]
        bin_slopes_for_plot = bin_slopes[plot_mask]
        bin_slope_errs_for_plot = bin_slope_errs[:,plot_mask]

        # Get the bin slope error by subtracting the bounds of the confidence intervals from the slope value
        #bin_slope_errs_for_plot = np.transpose(np.abs(bin_slope_errs_for_plot - bin_slopes_for_plot[:, np.newaxis]))


        # plot the bin fits
        plt.figure()
        plt.errorbar(ap_positions_for_plot, bin_slopes_for_plot, yerr=bin_slope_errs_for_plot, capsize=2, fmt='o')
        plt.xlabel('AP position')
        plt.ylabel('Fit rate of average trace (AU/min)')
        plt.title('Fit rate of average trace vs. AP position (with shifting)')
        plt.show()

        return ap_positions, bin_slopes, bin_slope_errs




