

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import datetime

class TraceFromRandomKernel:
    """
        This class simulates the expression of a gene with MS2 loops by generating a random kernel (representing
        the fluorescence profile of a single polymerase) and then convolving it with a provided initiation vector to produce
        a simulated fluorescence trace. Various noise models can be applied to the trace.

        Upgrades for the future:
            - Modular noise functions
            - Ability to drop timepoints

        Parameters
        ----------
        experiment_dt : int or float
            Time step between experimental measurements in seconds
        init_dt : int or float
            Time step between initiation configuration events in seconds
        ttotal : int
            Total simulation time in seconds
        nloops : int
            Number of MS2 stem loops in the reporter construct
        fmean : float
            Mean fluorescence per polymerase of the generated kernel
        fsigma : float
            Standard deviation of Gaussian distribution used to generate fluorescence value for each kernel
        tau1mean : float
            Mean rise time of kernel in seconds (time to transcribe the MS2 sequence)
        tau1sigma : float
            Standard deviation of rise time of kernel in seconds
        tau2mean : float
            Mean elongation time of kernel in seconds (time from MS2 region to end of gene)
        tau2sigma : float
            Standard deviation of elongation time of kernel in seconds
        initiationvec : array-like, optional
            Vector of initiation events (1 for initiation, 0 for no initiation)
        poisson_noise : bool, default=True
            Option to include Poisson noise in generated trace
        gaussian_noise : bool, default=False
            Option to include Gaussian noise in generated trace
        gaussian_noise_amplitude : float, default=1
            Amplitude of white noise if included (in same units as f - arbitrary units)
        epsilon : float, default=1
            Minimum error for each measurement
        noise_function : str, optional
            Optional parameter for time-correlated noise function.
            Options: "trig" or "heaviside_multiplicative". Defaults to None.

        Attributes
        ----------
        dt : int or float
            Time step between experimental measurements in seconds
        timepoints : ndarray
            Array of time points for experimental measurements
        init_timepoints : ndarray
            Array of time points for initiation events
        f : float
            Fluorescence per polymerase of the generated kernel
        tau1 : float
            Rise time of kernel in seconds
        tau2 : float
            Elongation time of kernel in seconds
        kernel : ndarray
            Generated fluorescence kernel
        raw_trace : ndarray
            Trace before noise addition
        correlated_noise_trace : ndarray
            Trace after adding correlated noise (if applicable)
        noisy_trace : ndarray
            Trace after all noise addition
        errorbars : ndarray
            Estimated uncertainties for each measurement
        offsets : str or ndarray
            Parameters of the noise function used (if applicable)
        """
    def __init__(self,experiment_dt = 3, init_dt = 1, ttotal = 1800, nloops = 24, fmean = 10, fsigma = 1, tau1mean = 50, tau1sigma = 5, tau2mean = 300, tau2sigma = 30, initiationvec = None, poisson_noise = True, gaussian_noise = False, gaussian_noise_amplitude = 1, epsilon = 1,noise_function = None):
        self.dt = experiment_dt
        self.init_dt = init_dt
        self.ttotal = ttotal
        self.nloops = nloops
        self.fmean = fmean
        self.fsigma = fsigma
        self.tau1mean = tau1mean
        self.tau2mean = tau2mean
        self.tau1sigma = tau1sigma
        self.tau2sigma = tau2sigma
        self.initiationvec = initiationvec
        self.poisson_noise = poisson_noise
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_amplitude = gaussian_noise_amplitude
        self.f = None
        self.tau1 = None
        self.tau2 = None
        self.raw_trace = None
        self.noisy_trace = None
        self.timepoints = np.arange(0, self.ttotal, self.dt)
        self.init_timepoints = np.arange(0, self.ttotal, self.init_dt)
        self.errorbars = None
        self.epsilon = epsilon
        self.noise_function = noise_function
        self.correlated_noise_trace = None
        self.offsets = None

    def choose_params(self):

        """
            Randomly select kernel parameters based on the specified distributions.

            Generates values for:
            - f: Fluorescence per polymerase
            - tau1: Rise time (time to transcribe MS2 loops)
            - tau2: Elongation time (time from loops to gene end)

            Values are drawn from normal distributions with means and standard deviations
            specified in the constructor. Negative values are clipped to zero.

            Returns
            -------
            None
                Updates the f, tau1, and tau2 attributes
            """

        self.f = np.max([0,np.random.normal(self.fmean, self.fsigma)])
     
        self.tau1= np.max([0,np.random.normal(self.tau1mean, self.tau1sigma)])
        self.tau2 = np.max([0,np.random.normal(self.tau2mean, self.tau2sigma)])
      

    def make_kernel(self):

        """
            Generate a fluorescence kernel based on the selected parameters.

            Creates a time-dependent fluorescence profile for a single polymerase,
            with a stepwise increase during the rise phase (transcription of MS2 loops)
            and a constant value during the elongation phase.

            Returns
            -------
            None
                Updates the kernel attribute
            """

        looptime = self.tau1/self.nloops
        times = np.zeros(int(self.ttotal))
        for i in range(len(times)):
            t_abs = i
            if t_abs <= self.tau1:
                times[i] = np.floor(t_abs/looptime)*self.f*1/self.nloops
            if self.tau1 < t_abs <=(self.tau1+self.tau2):
                times[i] = self.f

        self.kernel = times[::self.init_dt]





    def make_raw_trace(self):

        """
            Generate a raw fluorescence trace by convolving the kernel with the initiation vector.

            The raw trace represents the fluorescence signal before adding noise,
            created by convolving the initiation events with the fluorescence kernel.
            The resulting trace is interpolated to match the experimental timepoints.

            Returns
            -------
            None
                Updates the raw_trace attribute

            Raises
            ------
            Exception
                If convolution fails due to incompatible kernel and initiation vector
            """

        try:
            raw_trace = np.convolve(self.initiationvec, self.kernel, mode = 'full')

            sampled_raw_trace = np.interp(self.timepoints, self.init_timepoints, raw_trace[:int(self.ttotal/self.init_dt)])


            self.raw_trace = sampled_raw_trace
        except:
            print("Incompatible kernel and initiation vec - convolve failed")

    def add_correlated_noise(self):

        """
            Add time-correlated noise to the raw trace.

            Applies one of the specified noise models:
            - 'heaviside_multiplicative': Scales the trace by a random factor during a random time window
            - 'trig': Adds a sinusoidal offset to the trace

            Returns
            -------
            None
                Updates the correlated_noise_trace and offsets attributes
            """

        self.correlated_noise_trace = self.raw_trace
        correlated_trace = self.raw_trace

        if self.noise_function == 'heaviside_multiplicative':

            step = np.random.uniform(0, self.ttotal)

            step2 = np.random.uniform(step, self.ttotal)

            offsetscale = np.random.uniform(0.75, 1.25)

            self.offsets = str({'on':step, 'off':step2, 'scale': offsetscale})

            for i in range(len(self.raw_trace)):

                if step <= i*self.dt <step2:
                    correlated_trace[i] = correlated_trace[i]*offsetscale

            self.correlated_noise_trace = correlated_trace

        if self.noise_function == 'trig':

            offsetscale = np.random.uniform(0.25*self.f, 2*self.f)

            offsetperiod = np.random.uniform(0, self.ttotal)

            offsetfunction = offsetscale*np.sin(self.timepoints/offsetperiod)

            plt.figure()
            plt.plot(self.timepoints, offsetfunction)
            plt.show()

            self.offsets = offsetfunction

            self.correlated_noise_trace = np.sum([self.raw_trace, offsetfunction], axis = 0)

        self.correlated_noise_trace = [np.max([x,0]) for x in self.correlated_noise_trace]

    def make_noisy_trace(self):

        """
            Add time-correlated noise to the raw trace.

            Applies one of the specified noise models:
            - 'heaviside_multiplicative': Scales the trace by a random factor during a random time window
            - 'trig': Adds a sinusoidal offset to the trace

            Returns
            -------
            None
                Updates the correlated_noise_trace and offsets attributes
            """

        noisy_trace = self.correlated_noise_trace
        if self.poisson_noise:

            noisy_trace = [np.random.poisson(x) for x in noisy_trace]

        
        if self.gaussian_noise:

            noisy_trace = [np.random.normal(x, self.gaussian_noise_amplitude) for x in noisy_trace]

        

        self.noisy_trace = [np.max([x,0]) for x in noisy_trace]

    def get_errorbars_trace(self):
        """
            Calculate uncertainty estimates for each measurement in the trace.

            Computes error bars based on:
            - Minimum error (epsilon)
            - Poisson noise (square root of signal)
            - Gaussian noise amplitude

            Returns
            -------
            None
                Updates the errorbars attribute
            """

        errs = self.epsilon*np.ones(int(self.ttotal/self.dt))

        if self.poisson_noise:
            errs += np.array([np.sqrt(x) for x in self.noisy_trace])

        if self.gaussian_noise:
            errs += self.gaussian_noise_amplitude*np.ones(int(self.ttotal/self.dt))

        self.errorbars = errs



class InitiationsGenerator:
    """
    Class to generate initiation vectors for different promoter models.

    This class simulates transcriptional initiation events based on different promoter
    activity models. It can generate initiation vectors that represent the timing of
    polymerase loading onto a gene.

    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducible initiations
    niters_init : int, default=1
        Number of initiation configurations to generate
    ttotal : int, default=1800
        Total time in seconds
    init_dt : float or int, default=1
        Spacing between timepoints in seconds
    promoter_model : str, default='uniform'
        Promoter model to use. Options are:
        - 'uniform': Uniform probability of initiation at each time step
        - 'constitutive': Exponential waiting time distribution
        - 'twostate': Two-state promoter model (on/off switching)
        - 'threestate': Represents chromatids - sum of 2 two-state models,
          each promoter transcribes at rate r
    p_uniform : float, default=0.33
        Probability (in 1/s) of initiations for uniform model.
        Note: p_uniform*dt should be <=1
    k_on : float, default=1/60.
        Rate of switching from OFF to ON state for 2 and 3 state bursting models.
        Note: k_on*dt should be <1 to avoid artifacts
    k_off : float, default=0.7/60
        Rate of switching from ON to OFF state for 2 and 3 state bursting models.
        Note: k_off*dt should be <1 to avoid artifacts
    r : float, default=8/60
        Initiation rate once the promoter is in the ON state.
        Also used for constitutive promoter, which will be k_on = 1 and k_off = 0
    mitotic_repression : bool, default=True
        Optional silencing of initiations for the ending of the simulation
        to represent mitotic repression
    repression_window : int, default=300
        Amount of time for mitotic repression in seconds, applied to the
        end of the simulation

    Attributes
    ----------
    model : str
        Selected promoter model

    Methods
    -------
    genwaitingtime(mu)
        Generates exponentially distributed waiting time with mean mu
    gen_burst_arrival_times(mean, burststart, burststop)
        Generates arrival times for initiations within a burst
    burstgen()
        Generates ON/OFF burst windows using stochastic simulation
    gen_loading_times(tons, toffs)
        Generates polymerase loading times during ON states
    gen_bursts_and_loading_times()
        Generates both burst windows and loading times
    make_initiation_config()
        Creates a complete initiation vector based on the selected model
    simulate_config_dataframe()
        Generates multiple initiation configurations and returns as dataframe
    """

    def __init__(self, seed = 42, niters_init = 1, ttotal = 1800, init_dt = 1, promoter_model = 'uniform', p_uniform = 0.33, k_on = 1/60., k_off = 0.7/60, r = 8/60, mitotic_repression = True, repression_window = 300):

        self.seed = seed
        self.niters = niters_init
        self.ttotal = ttotal
        self.dt = init_dt
        self.model = promoter_model
        self.p_uniform = p_uniform
        self.k_on = k_on
        self.k_off = k_off
        self.r = r
        self.mitotic_repression = mitotic_repression
        self.repression_window = repression_window


    def genwaitingtime(self, mu):
        """
        Generate an exponentially distributed waiting time.

        Parameters
        ----------
        mu : float
            Mean waiting time

        Returns
        -------
        float
            Random waiting time drawn from exponential distribution
        """

        p = np.random.random()

        tp = -mu * np.log(1 - p)
        return tp

    def gen_burst_arrival_times(self, mean, burststart, burststop):
        """
            Generate arrival times for initiations within a transcriptional burst.

            Parameters
            ----------
            mean : float
                Mean waiting time between initiations
            burststart : float
                Start time of the burst
            burststop : float
                End time of the burst

            Returns
            -------
            list
                List of initiation times within the burst
            """

        arrivaltimes = []

        while len(arrivaltimes) == 0:
            newtime = self.genwaitingtime(mean)
            arrivaltimes.append(newtime + burststart)

        while arrivaltimes[-1] < (burststop - self.dt):
            newtime = self.genwaitingtime(mean)
            arrivaltimes.append(newtime + arrivaltimes[-1])

        return arrivaltimes[:-1]

    def burstgen(self):
        """
            Generate burst windows using stochastic simulation of a two-state system.

            Simulates the promoter switching between ON and OFF states according to
            the specified k_on and k_off rates.

            Returns
            -------
            ndarray
                Array of times when the promoter switches to ON state
            ndarray
                Array of times when the promoter switches to OFF state
            """
        burst = 0
        simtimes = np.arange(0, self.ttotal, self.dt)
        outputs = np.zeros(len(simtimes))

        for dt in range(int(self.ttotal/self.dt)):

            outputs[dt] = burst
            p = np.random.random()

            if burst == 0:
                if p < self.k_on*self.dt:
                    burst = 1
                continue

            elif burst > 0:
                if p < self.k_off*self.dt:
                    burst = 0


        der = np.gradient(outputs)

        on_indexes = np.where(der > 0)
        tons = simtimes[on_indexes][::2]

        off_indexes = np.where(der < 0)
        toffs = simtimes[off_indexes][::2]

        while len(toffs) < len(tons):
            toffs = np.append(toffs, simtimes[-1])

        return tons, toffs

    def gen_loading_times(self, tons, toffs):
        """
            Generate polymerase loading times during burst windows.

            Parameters
            ----------
            tons : ndarray
                Array of times when the promoter switches to ON state
            toffs : ndarray
                Array of times when the promoter switches to OFF state

            Returns
            -------
            list
                List of polymerase loading times
            """

        finalarrivals = []

        mean  = 1/self.r

        for i in range(len(tons)):
            # print('start: '+str(tons[i])+ ' stop:' +str(toffs[i]))

            arrivals = self.gen_burst_arrival_times(mean, tons[i], toffs[i])

            finalarrivals.extend(arrivals)

        return finalarrivals

    def gen_bursts_and_loading_times(self):
        """
            Generate both burst windows and polymerase loading times.

            Returns
            -------
            list
                List of polymerase loading times
            ndarray
                Array of times when the promoter switches to ON state
            ndarray
                Array of times when the promoter switches to OFF state
            """

        tons, toffs = self.burstgen()

        loadtimes = self.gen_loading_times(tons, toffs)

        return loadtimes, tons, toffs

    def make_initiation_config(self):
        """
            Create a complete initiation vector based on the selected promoter model.

            Implements different promoter models:
            - uniform: Constant probability of initiation at each time step
            - constitutive: Special case of two-state with k_on=1, k_off=0
            - twostate: Bursting model with ON/OFF switching
            - threestate: Two independent two-state models (representing chromatids)

            Returns
            -------
            ndarray
                Binary initiation vector (1 for initiation, 0 for no initiation)
            list or None
                Burst windows information
            dict
                Parameters used for the initiation model
            """

        initiation_vector = None
        burst_window = None
        init_params = None

        if self.model == 'uniform':

            initiation_vector = np.random.binomial(1, self.p_uniform*self.dt, int(self.ttotal/self.dt))
            burst_windows = None
            init_params = {'model': 'uniform', 'p_uniform': self.p_uniform}

        elif self.model == 'constitutive':
            self.k_on = 1
            self.k_off = 0
            init_params = {'model':'constitutive','r':self.r}
            self.model = 'twostate'

        if self.model == 'twostate':

            loadtimes, tons, toffs = self.gen_bursts_and_loading_times()

            initiation_vector = np.zeros(int(self.ttotal / self.dt))

            for i in range(int(self.ttotal/self.dt)):
                for j in loadtimes:
                    if i*self.dt <= j < (i+1)*self.dt:
                        initiation_vector[i] = 1


            burst_windows = [tons, toffs]
            if init_params == None:
                init_params = {'model':'twostate', 'k_on': self.k_on, 'k_off': self.k_off, 'r': self.r}


        if self.model == 'threestate':

            self.r = 0.5*self.r

            loadtimes1, tons1, toffs1 = self.gen_bursts_and_loading_times()
            loadtimes2, tons2, toffs2 = self.gen_bursts_and_loading_times()

            initiation_vector = np.zeros(int(self.ttotal / self.dt))

            for i in range(int(self.ttotal / self.dt)):
                for j in loadtimes1:
                    if i * self.dt <= j < (i + 1) * self.dt:
                        initiation_vector[i] = 1


                for k in loadtimes2:
                    if i * self.dt <= k < (i + 1) * self.dt:
                        initiation_vector[i] += 1

            burst_windows = [[tons1, toffs1],[tons2, toffs2]]
            init_params = {'model':'threestate', 'k_on': self.k_on, 'k_off': self.k_off, 'r': self.r}

        return initiation_vector, burst_windows, init_params

    def simulate_config_dataframe(self):
        """
               Generate multiple initiation configurations and return as a dictionary.

               Uses the specified seed for reproducibility and applies mitotic repression
               if enabled.

               Returns
               -------
               dict
                   Dictionary containing:
                   - 'trueIvec': List of initiation vectors
                   - 'bursts': List of burst window information
                   - 'init_params': List of initiation model parameters
               """
        np.random.seed(self.seed)

        bursts = []
        configs = []
        init_params = []

        for i in range(self.niters):

            config, burstvals, params = self.make_initiation_config()
            if self.mitotic_repression:
                r = int(self.repression_window/self.dt)

                config[-r:] = 0

            configs.append(config)
            bursts.append(burstvals)
            init_params.append(params)

        initiation_df = {'trueIvec':configs, 'bursts':bursts, 'init_params':init_params}

        return initiation_df

class FullTraceSimulator:
    """
    Combines the InitiationsGenerator and TraceFromRandomKernel classes to create
    complete simulated fluorescence traces.

    This class handles the creation of initiation vectors and generates multiple
    fluorescence traces with different kernels for each initiation configuration.
    It can save the parameters and results to disk.

    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducible results
    niters_experiment : int, default=10
        Number of kernel variations to generate for each initiation configuration
    niters_init : int, default=1
        Number of initiation configurations to generate
    filepath : str, optional
        Path to save simulation results
    param_ftype : str, default='pkl'
        File type for saving parameters ('pkl' or 'txt')
    ttotal : int, default=1800
        Total simulation time in seconds
    init_dt : int, default=1
        Time step between initiation events in seconds
    promoter_model : str, default='uniform'
        Promoter model to use (see InitiationsGenerator)
    p_uniform : float, default=0.33
        Probability of initiation for uniform model
    k_on : float, default=1/60.
        Rate of switching to ON state for bursting models
    k_off : float, default=0.7/60
        Rate of switching to OFF state for bursting models
    r : float, default=8/60
        Initiation rate in ON state
    mitotic_repression : bool, default=True
        Whether to include mitotic repression
    repression_window : int, default=300
        Duration of mitotic repression window in seconds
    experiment_dt : int, default=3
        Time step between experimental measurements in seconds
    nloops : int, default=24
        Number of MS2 stem loops
    fmean : float, default=10
        Mean fluorescence per polymerase
    fsigma : float, default=1
        Standard deviation of fluorescence per polymerase
    tau1mean : float, default=50
        Mean rise time in seconds
    tau1sigma : float, default=5
        Standard deviation of rise time in seconds
    tau2mean : float, default=300
        Mean elongation time in seconds
    tau2sigma : float, default=30
        Standard deviation of elongation time in seconds
    poisson_noise : bool, default=True
        Whether to include Poisson noise
    gaussian_noise : bool, default=False
        Whether to include Gaussian noise
    gaussian_noise_amplitude : float, default=1
        Amplitude of Gaussian noise
    epsilon : float, default=1
        Minimum error for each measurement
    noise_function : str, optional
        Function for time-correlated noise

    Attributes
    ----------
    init_params : dict
        Dictionary of parameters for the InitiationsGenerator
    trace_params : dict
        Dictionary of parameters for the TraceFromRandomKernel
    combined_df : DataFrame
        DataFrame containing all simulation results

    Methods
    -------
    saveparams()
        Saves simulation parameters to disk
    simulate_one_config(init_dict, index)
        Simulates multiple traces for one initiation configuration
    simulate()
        Runs the complete simulation
    save_results()
        Saves simulation results to disk
    load_results(runID=None)
        Loads saved simulation results
    """
    def __init__(self,
                 seed = 42,
                 niters_experiment = 10,
                 niters_init = 1,
                 filepath = None,
                 store_params = True,
                 param_ftype = 'pkl',
                 ttotal = 1800,
                 init_dt = 1,
                 promoter_model = 'uniform',
                 p_uniform = 0.33,
                 k_on = 1/60.,
                 k_off = 0.7/60,
                 r = 8/60,
                 mitotic_repression = True,
                 repression_window = 300,
                 experiment_dt = 3,
                 nloops = 24,
                 fmean = 10,
                 fsigma = 1,
                 tau1mean = 50,
                 tau1sigma = 5,
                 tau2mean = 300,
                 tau2sigma = 30,
                 poisson_noise = True,
                 gaussian_noise = False,
                 gaussian_noise_amplitude = 1,
                 epsilon = 1,
                 noise_function = None):

        self.seed = seed
        self.niters_experiment = niters_experiment
        self.niters_init = niters_init
        self.filepath = filepath
        self.param_ftype = param_ftype
        self.ttotal = ttotal
        self.init_dt = init_dt
        self.promoter_model = promoter_model
        self.p_uniform = p_uniform
        self.k_on = k_on
        self.k_off = k_off
        self.r = r
        self.mitotic_repression = mitotic_repression
        self.repression_window = repression_window
        self.experiment_dt = experiment_dt
        self.nloops = nloops
        self.fmean = fmean
        self.fsigma = fsigma
        self.tau1mean = tau1mean
        self.tau1sigma = tau1sigma
        self.tau2mean = tau2mean
        self.tau2sigma = tau2sigma
        self.poisson_noise = poisson_noise
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_amplitude = gaussian_noise_amplitude
        self.epsilon = epsilon
        self.noise_function = noise_function
        self.init_params = dict(seed = self.seed,
                                niters_init = self.niters_init,
                                ttotal = self.ttotal,
                                init_dt = self.init_dt,
                                promoter_model = self.promoter_model,
                                p_uniform = self.p_uniform,
                                k_on = self.k_on,
                                k_off = self.k_off,
                                r = self.r,
                                mitotic_repression = self.mitotic_repression,
                                repression_window = self.repression_window)
        self.trace_params = dict(experiment_dt = self.experiment_dt,
                 nloops = self.nloops,
                 fmean = self.fmean,
                 fsigma = self.fsigma,
                 tau1mean = self.tau1mean,
                 tau1sigma = self.tau1sigma,
                 tau2mean = self.tau2mean,
                 tau2sigma = self.tau2sigma,
                 poisson_noise = self.poisson_noise,
                 gaussian_noise = self.gaussian_noise,
                 gaussian_noise_amplitude = self.gaussian_noise_amplitude,
                 epsilon = self.epsilon,
                 noise_function = self.noise_function)

        self.combined_df = None

    def saveparams(self):
        """
            Save simulation parameters to disk.

            Creates a directory if it doesn't exist and saves parameters as a pickle file
            or text file depending on param_ftype.

            Returns
            -------
            None

        """
        all_params = {'timestamp':str(datetime.datetime.now()), 'init_params':self.init_params,'trace_params':self.trace_params, 'niters_experiment': self.niters_experiment}
        fplen = len(self.filepath)
        if os.path.isfile(self.filepath+'/params.pkl') or os.path.isfile(self.filepath+'/params.txt') or os.path.isfile(self.filepath+'/traces.pkl'):
            i = 000
            newfpath = self.filepath[:fplen] + '_'+str(i)
            while os.path.isfile(newfpath + '/params.pkl'):
                i += 1
                newfpath = self.filepath[:fplen] +'_'+ str(i)
            self.filepath = newfpath

        os.mkdir(self.filepath) if not os.path.exists(self.filepath) else None

        if self.saveparams:


            if self.param_ftype == 'pkl':

                with open(self.filepath+str('/params.pkl'), 'wb') as f:
                    pickle.dump(all_params, f)

                #print('Params saved in {}'.format(self.filepath+str('/params.pkl')))

            else:
                with open(self.filepath+str('/params.txt'), 'w') as f:
                    f.write(str(all_params))

                #print('Params saved in {}'.format(self.filepath+str('/params.txt')))

        #if not self.saveparams:
            #print('No parameters saved.')

        return

    def simulate_one_config(self, init_dict, index):
        """
            Simulate multiple traces for one initiation configuration.

            Generates niters_experiment different kernel variations for a single
            initiation configuration.

            Parameters
            ----------
            init_dict : dict
                Dictionary containing initiation vectors and parameters
            index : int
                Index of the initiation configuration to use

            Returns
            -------
            DataFrame
                DataFrame containing all simulation results for this configuration
            """

        np.random.seed(self.seed)
        fs = np.zeros(self.niters_experiment)
        tau1s = np.zeros(self.niters_experiment)
        tau2s = np.zeros(self.niters_experiment)
        raw_traces = []
        noise_traces = []
        timevecs = []
        errs = []
        offsets = []
        ivecs = []
        bursts = []
        init_p = []

        for i in range(self.niters_experiment):
            sample = TraceFromRandomKernel(**self.trace_params, init_dt = self.init_dt, ttotal = self.ttotal, initiationvec = init_dict['trueIvec'][index])

            sample.choose_params()
            sample.make_kernel()
            sample.make_raw_trace()
            sample.add_correlated_noise()
            sample.make_noisy_trace()
            sample.get_errorbars_trace()

            fs[i] = sample.f
            tau1s[i] = sample.tau1
            tau2s[i] = sample.tau2
            raw_traces.append(sample.raw_trace)
            noise_traces.append(sample.noisy_trace)
            timevecs.append(sample.timepoints)
            errs.append(sample.errorbars)
            offsets.append(sample.offsets)
            ivecs.append(init_dict['trueIvec'][index])
            bursts.append(init_dict['bursts'][index])
            init_p.append(init_dict['init_params'][index])

        resultsdict = {"trueIvec":ivecs, "bursts":bursts, "initiation_params":init_p,"f_per_pol": fs, "rise_time": tau1s, "ss_time": tau2s, "raw_trace": raw_traces,
                              "noise_added_trace": noise_traces, "uncertainties": errs,
                              "timepoints": timevecs, "offsets": offsets}
        df = pd.DataFrame(resultsdict)
        return df

    def simulate(self):
        """
            Run the complete simulation.

            Generates initiation configurations and kernel variations, and combines
            the results into a DataFrame.

            Returns
            -------
            None
                Updates the combined_df attribute
            """

        ivecgen = InitiationsGenerator(**self.init_params)

        init_results = ivecgen.simulate_config_dataframe()
        dfs = []

        for i in range(self.niters_init):

            kernel_sample_df = self.simulate_one_config(init_results, i)
            dfs.append(kernel_sample_df)

        self.combined_df = pd.concat(dfs, ignore_index = True)

    def save_results(self):

        """
            Save simulation results to disk.

            Saves parameters and the combined DataFrame to the specified filepath.

            Returns
            -------
            None
            """

        self.saveparams()

        with open(self.filepath+str('/traces.pkl'), 'wb') as f:
                pickle.dump(self.combined_df, f)

        print('Trace saved in {}'.format(self.filepath+str('/traces.pkl')))

    def load_results(self):

        """
            Load saved simulation results.


            Returns
            -------
            DataFrame
                DataFrame containing trace simulation results
            dict or str
                Parameters used for the simulation
            """

        print('Opening files from path {}'.format(self.filepath))

        with open(self.filepath+str('/traces.pkl'), 'rb') as f:
            results_df = pickle.load(f)

        if self.param_ftype == 'pkl':
            with open(self.filepath+str('/params.pkl'), 'rb') as f:
                params_df = pickle.load(f)
        else:
            with open(self.filepath+str('/params.txt'), 'r') as f:
                params_df = f.read()

        return results_df, params_df

