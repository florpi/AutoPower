"""
Generate the power spectra for a given parameter space in parallel and
store the result in an HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import copy
import h5py
import numpy as np
import os
import pandas as pd
import sys
import time
import warnings

from multiprocessing import Process
from tqdm import tqdm

from autopower.data.datageneration import get_power_spectrum
from autopower.utils.multiprocessing import Queue

from autopower.utils import latinhypercube as lh


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def queue_worker(parameters, results_queue):
    """
    Helper function to generate a single sample in a dedicated process.

    Args:
        parameters (dict): Dictionary containing the parameters that
            are passed to get_power_spectrum().
        results_queue (Queue): The queue to which the results of this
            worker / process are passed.
    """

    # Try to generate a sample using the given arguments and store the result
    # in the given result_queue (which is shared across all worker processes).
    try:

        # Run the simulation and collect the results
        kh, pk_lin = get_power_spectrum(**parameters, mode='linear')
        kh, pk_nonlin = get_power_spectrum(**parameters, mode='non-linear')

        # Combine the results with their parameters into a single result dict
        result = copy.deepcopy(parameters)
        result['kh'] = kh
        result['pk_ratio'] = pk_nonlin/pk_lin

        # Add results to the result queue and exit with exit code zero
        results_queue.put(result)
        sys.exit(0)

    # In case the simulation does not succeed for some reason, catch the error
    # and return with a non-zero exit code.
    except RuntimeError:
        sys.exit('Runtime Error')


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Start the stopwatch
    script_start = time.time()

    print('')
    print('GENERATE POWER SPECTRA')
    print('')
    
    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------
    
    # Set up the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-runtime',
                        help='Maximum runtime per process in seconds '
                             '(default: 60).',
                        type=int,
                        default=60)
    parser.add_argument('--n-processes',
                        help='Number of parallel processes (default: 1).',
                        type=int,
                        default=1)
    parser.add_argument('--no-progressbar',
                        help='Do not show a progressbar while generating '
                             'samples.',
                        action='store_true')
    parser.add_argument('--random-seed',
                        help='Random seed for numpy (default: 42).',
                        type=int,
                        default=42)
    parser.add_argument('--use-pandas',
                        help='Use pandas to save the results to an HDF file.',
                        action='store_true')
    parser.add_argument('--n-samples',
                        help='Number of parameter combinations for which to '
                             'compute the power spectrum (default: 100).',
                        type=int,
                        default=100)

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    args = parser.parse_args()
    print('Done!')

    # Set the random seed for this script
    np.random.seed(args.random_seed)

    # -------------------------------------------------------------------------
    # Define the parameter space for which to generate power spectra
    # -------------------------------------------------------------------------

    '''
    h_values = np.linspace(0.5, 0.7, 4)
    omc_values = np.linspace(0.24, 0.26, 4)

    parameter_combinations = list()
    for h, omc in itertools.product(h_values, omc_values):
        parameter_combinations.append(dict(h=h, omc=omc))
    '''

    # TODO: This should be made more descriptive (i.e., what parameter does
    #       the minimum / maximum belong to?
    minimum_values = [0.65, 0.10]
    maximum_values = [0.69,  0.13]

    # Use a latin hypercube to generate the parameter combinations for which
    # we will generate a sample
    parameter_combinations = \
        lh.latin_hypercube(args.n_samples, minimum_values, maximum_values)
    parameter_combinations = \
        [{'h': h, 'omc': omc} for h, omc in zip(parameter_combinations[:, 0],
                                                parameter_combinations[:, 1])]

    # -------------------------------------------------------------------------
    # Create samples (i.e., simulate power spectra for these combinations)
    # -------------------------------------------------------------------------

    print('\nGenerating samples:', flush=True)

    # Initialize a Queue and fill it with all parameter combinations for
    # which we want to simulate the power spectrum
    arguments_queue = Queue()
    for parameter_combination in parameter_combinations:
        arguments_queue.put(parameter_combination)

    # Initialize a Queue and a list to store the generated samples
    results_queue = Queue()
    results_list = []

    # Set up the context manager based on whether or not we show a progressbar
    if args.no_progressbar:
        # This is a simple workaround for a no-op context manager, since
        # contextlib.nullcontext() is only available for Python 3.7+.
        context = memoryview(b'')
    else:
        tqdm_args = dict(total=args.n_samples, ncols=80, unit='sample')
        context = tqdm(**tqdm_args)

    # Use process-based multiprocessing to generate samples in parallel
    with context as progressbar:

        # Keep track of all running processes
        list_of_processes = []

        # Keep going as long as there are still parameter combinations we
        # have not yet processed, or if there are still processes running
        while arguments_queue.qsize() > 0 or len(list_of_processes):

            # Loop over all processes to see if anything finished or got stuck
            for process_dict in list_of_processes:

                # Get the process object and its current runtime
                process = process_dict['process']
                runtime = time.time() - process_dict['start_time']

                # Check if the process is still running when it should
                # have terminated already (according to max_runtime)
                if process.is_alive() and (runtime > args.max_runtime):

                    # Kill process that's been running too long
                    process.terminate()
                    process.join()
                    list_of_processes.remove(process_dict)

                    # Print a warning when this happens
                    warnings.warn('A process exceed the maximum runtime and '
                                  'was killed!')

                # If process has terminated already
                elif not process.is_alive():

                    # Remove process from the list of running processes
                    list_of_processes.remove(process_dict)

            # Start new processes until the arguments_queue is empty, or
            # we have reached the maximum number of processes
            while (arguments_queue.qsize() > 0 and
                   len(list_of_processes) < args.n_processes):

                # Get arguments from queue and start new process
                parameters = arguments_queue.get()
                p = Process(target=queue_worker,
                            kwargs=dict(parameters=parameters,
                                        results_queue=results_queue))

                # Remember this process and its starting time
                process_dict = dict(process=p, start_time=time.time())
                list_of_processes.append(process_dict)

                # Finally, start the process
                p.start()

            # Move results from results_queue to results_list (without
            # this part, the results_queue blocks the worker processes
            # so that they won't terminate!)
            while results_queue.qsize() > 0:
                results_list.append(results_queue.get())

            # Update the progress bar based on the number of results
            if not args.no_progressbar:
                progressbar.update(len(results_list) - progressbar.n)

            # Sleep for some time before we check the processes again
            time.sleep(0.1)

    print('Sample generation completed!\n', flush=True)

    # -------------------------------------------------------------------------
    # Save the results to an HDF file
    # -------------------------------------------------------------------------

    print('Saving the results to HDF file ...', end=' ', flush=True)

    # Construct a pandas data frame from the results
    dataframe = pd.DataFrame(results_list)

    # Define a file path where to save the results
    hdf_file_path = '../data/results.hdf'

    # If requested, save the data frame directly using pandas
    if args.use_pandas:
        dataframe.to_hdf(hdf_file_path, key='data_as_pandas_dataframe')

    # Otherwise, simply use a vanilla HDF file
    else:
        # TODO: This needs to be adjusted to the parameters we are using!
        with h5py.File(hdf_file_path, 'w') as hdf_file:
            for key in ('kh', 'pk_ratio', 'h', 'omc'):
                data = dataframe[key].to_numpy()
                if key in ('kh', 'pk_ratio'):
                    data = np.row_stack(data)
                hdf_file.create_dataset(name=key, data=data, shape=data.shape)

    print('Done!', flush=True)
    
    # Get file size in MB and print the result
    sample_file_size = os.path.getsize(hdf_file_path) / 1024 ** 2
    print('Size of resulting HDF file: {:.2f}MB'.format(sample_file_size))
    print('')
    
    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # Print the total run time
    print('Total runtime: {:.1f} seconds!'.format(time.time() - script_start))
    print('')
