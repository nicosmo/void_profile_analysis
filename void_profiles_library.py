
import numpy as np
import h5py
import os
import multiprocessing as mp


# --- Conditional Import for VIDE or SciPy ---
try:
    #
    from vide import periodic_kdtree as pkdtree
    VIDE_AVAILABLE = True
    print("VIDE library found. Using optimized vide.periodic_kdtree.PeriodicCKDTree.")
    #
except ImportError:
    #
    from scipy.spatial import cKDTree
    VIDE_AVAILABLE = False
    print("VIDE library not found. Falling back to scipy.spatial.cKDTree.")
    print("For potentially better performance on large datasets, consider installing VIDE.")
    #




# --- Core Calculation & Stacking Logic ---

def _jackknife(profiles_in_bin, mean_profile):
    """
    Performs Jackknife resampling to estimate the covariance matrix of a set of profiles.

    This method estimates the variance and covariance of a stacked profile by
    iteratively leaving out one profile at a time from the sample, recalculating
    the mean, and measuring the variance of these "leave-one-out" means.

    Args:
        profiles_in_bin (np.ndarray): A 2D array of shape (n_voids, n_bins)
            containing all individual profiles within a single stack.
        mean_profile (np.ndarray): A 1D array of shape (n_bins,) representing
            the simple mean of all profiles in `profiles_in_bin`.

    Returns:
        np.ndarray: The estimated covariance matrix of shape (n_bins, n_bins).
    """
    n_voids = len(profiles_in_bin)
    if n_voids <= 1:
        return np.zeros((profiles_in_bin.shape[1], profiles_in_bin.shape[1]))
    #
    # Calculate all N leave-one-out means at once for efficiency
    total_sum = np.sum(profiles_in_bin, axis=0)
    leave_one_out_means = (total_sum - profiles_in_bin) / (n_voids - 1)
    # The jackknife samples are the differences from the total mean
    jackknife_samples = leave_one_out_means - mean_profile
    # Calculate the final covariance matrix
    covariance = (n_voids - 1.)**2 / n_voids * np.cov(jackknife_samples, rowvar=0)
    return covariance



def _calc_one_profile(i, profile_type, posVoid, radVoid, posTracer, velTracer,
                      tracerWeights, tree, nz, Lbox, rmax, Nbin, periodic, meanWeight):
    """
    Calculates a single radial profile for a specific void i.

    This is an internal helper function designed to be called in parallel by a
    wrapper function like `calculate_and_save_individual_profiles`. It finds all
    tracers within a given radius around a void, bins them, and computes the requested
    profile type.

    See https://arxiv.org/abs/2210.02457 for more details.

    This function consistently returns raw, un-normalized sums for all weighted profile
    types to ensure components can be correctly stacked. The correct normalization 
    is applied when the profiles are stacked.

    Args:
        i (int): The index of the void in the `posVoid` and `radVoid` arrays to be processed.
        profile_type (str): The type of profile to calculate. Supported options are:
                            'number_density', 'volume_weighted', 'mass_weighted', and 'velocity'.
        posVoid (np.ndarray): (N, 3) array of all void positions.
        radVoid (np.ndarray): (N,) array of all void radii.
        posTracer (np.ndarray): (M, 3) array of all tracer positions.
        velTracer (np.ndarray): (M, 3) array of all tracer velocities.
        tracerWeights (np.ndarray): (M,) array of weights for each tracer (e.g., volume or mass).
        tree (object): A pre-computed k-d tree for efficient neighbor searching. Can be
                       either a VIDE PeriodicCKDTree or a SciPy cKDTree.
        nz (float): The mean number density of the tracers in the simulation box.
        Lbox (float): The side length of the simulation box, used for periodic boundaries.
        rmax (int): The maximum radius for the profile in units of the void's radius (R_v).
        Nbin (int): The number of radial bins to use in the profile up to rmax.
        periodic (bool): If True, periodic boundary conditions are applied.
        meanWeight (float): The pre-calculated mean of the `tracerWeights` array. Required
                            for 'volume_weighted' and 'mass_weighted' profiles.

    Returns:
        np.ndarray: A 1D NumPy array of length `Nbin` containing the calculated profile.
                    Returns an array of zeros if an error (e.g., zero-volume shell) occurs.
    """
    #
    posV = posVoid[i, :]
    radius = radVoid[i]
    # Radius of the sphere around void i within which the profile is calculated
    r_query = rmax * radius 
    r_steps = np.linspace(0, rmax, Nbin + 1)
    #
    # Query the tree to find neighboring tracers
    if VIDE_AVAILABLE and periodic:
        indices = tree.query_ball_point(posV, r_query)
    else: # SciPy cKDTree
        indices_replicated = tree.query_ball_point(posV, r_query)
        # Map replicated indices back to original indices IF periodic
        if periodic:
            indices = np.unique(np.array(indices_replicated) % len(posTracer))
        else:
            indices = np.array(indices_replicated)
    #
    # Select the relevant tracers
    posT = posTracer[indices]
    velT = velTracer[indices]
    weightsT = tracerWeights[indices]
    # Calculate distances of tracers from void center, handling periodic boundaries
    dx = posT - posV
    if periodic:
        dx = (dx + Lbox / 2) % Lbox - Lbox / 2
    #
    dx_norm_phys = np.sqrt(np.sum(dx**2, axis=1))
    # Normalize distances by void radius
    dx_norm_rv = dx_norm_phys / radius
    #
    profile = np.zeros(Nbin)
    #
    if profile_type == 'number_density':
        # Calculate the physical volume of the shells for this specific void
        shell_volumes_phys = 4.0/3.0 * np.pi * (r_steps[1:]**3 - r_steps[:-1]**3) * radius**3
        # Prevent division by zero
        if np.any(shell_volumes_phys == 0):
            print(f"Error in calculating profile of void {i}: A shell volume is zero.")
            return profile
        #
        counts, _ = np.histogram(dx_norm_rv, bins=r_steps)
        profile = (counts / shell_volumes_phys) / nz
        #
    elif profile_type in ['volume_weighted', 'mass_weighted']:
        # Return the raw sum of weights, which is the denominator for velocity profiles.
        profile, _ = np.histogram(dx_norm_rv, bins=r_steps, weights=weightsT)
        #
    elif profile_type == 'velocity':
        #
        safe_dx_norm = np.where(dx_norm_phys == 0, 1, dx_norm_phys)
        v_radial = np.sum(dx * velT, axis=1) / safe_dx_norm
        v_radial[dx_norm_phys == 0] = 0
        profile, _ = np.histogram(dx_norm_rv, bins=r_steps, weights=v_radial * weightsT)
        #
    #
    return profile



# --- Global variable for multiprocessing pool initializer ---
_worker_init_args = {}



def _init_worker_pool(args):
    """
    Initializer for each worker in the multiprocessing pool.

    This is a key performance optimization. It is called once per
    worker process upon its creation and loads large, read-only data objects
    into that worker's memory space, avoiding repeated serialization.

    Args:
        args (dict): A dictionary containing the large data objects to be
                     shared with all worker processes.
    """
    global _worker_globals
    _worker_globals = args



def _parallel_task_wrapper(i_and_ptype):
    """
    A simple wrapper that acts as the target function for `multiprocessing.Pool.map`.

    The `pool.map` method can only iterate over a single argument. This function
    takes a tuple containing the task-specific arguments (`i` and `profile_type`)
    and unpacks them. It then calls the main calculation function, `_calc_one_profile`,
    passing both the task-specific arguments and the large, shared data arrays
    that were pre-loaded into the `_worker_globals` dictionary by the initializer.

    Args:
        i_and_ptype (tuple): A tuple containing the void index and the profile type string,
                             e.g., (10, 'number_density').

    Returns:
        np.ndarray: The calculated 1D profile array returned by `_calc_one_profile`.
    """
    i, profile_type = i_and_ptype
    return _calc_one_profile(i, profile_type, **_worker_globals)



def _build_scipy_periodic_kdtree(posTracer, Lbox):
    """
    Replicates tracer data in 26 neighboring boxes to simulate periodic boundaries for SciPy's cKDTree.
    """
    replicated_tracers = [posTracer]
    for dx in [-Lbox, 0, Lbox]:
        for dy in [-Lbox, 0, Lbox]:
            for dz in [-Lbox, 0, Lbox]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                #
                replicated_tracers.append(posTracer + np.array([dx, dy, dz]))
                #
            #
        #
    #
    return cKDTree(np.vstack(replicated_tracers))



# --- Function to calculate and save all individual profiles ---

def calculate_and_save_individual_profiles(save_path, posVoid, radVoid, posTracer, velTracer,
    tracerWeights, Lbox, profile_types, voidIDs=None, rmax=5, N_radial_bins=50, N_cpus=8, periodic=True ):
    """
    Calculates individual void profiles for all voids and saves them to an HDF5 file.

    Args:
        save_path (str): The full path for the output HDF5 file.
        posVoid (np.ndarray): (N, 3) array of void positions.
        radVoid (np.ndarray): (N,) array of void radii.
        posTracer (np.ndarray): (M, 3) array of tracer positions.
        velTracer (np.ndarray): (M, 3) array of tracer velocities.
        tracerWeights (np.ndarray): (M,) array of tracer weights (e.g., volume).
        Lbox (float): Side length of the periodic simulation box.
        profile_types (list): A list of profile types to calculate (e.g., ['number_density', 'velocity']).
        voidIDs (np.ndarray, optional): Array of void IDs. If None, sequential IDs will be generated.
        rmax (int): Max radius in units of void radii.
        N_radial_bins (int): Number of radial bins in each profile.
        N_cpus (int): Number of CPUs to use.
        periodic (bool, optional): If True, handles periodic boundaries. 
                                   If False, assumes standard Euclidean space. 
                                   Defaults to True.
    """
    # If no void IDs are provided, generate a default sequence
    if voidIDs is None:
        print("`voidIDs` not provided. Generating sequential IDs.")
        voidIDs = np.arange(len(posVoid))
    #
    # Creating the global tree of tracers
    if periodic:
        if VIDE_AVAILABLE:
            print("Building periodic tree with VIDE backend...")
            tree = pkdtree.PeriodicCKDTree(np.array([Lbox, Lbox, Lbox]), posTracer)
        else:
            print("Building periodic tree with SciPy backend (data replication)...")
            tree = _build_scipy_periodic_kdtree(posTracer, Lbox)
    else: # Non-periodic case
        if not VIDE_AVAILABLE:
            from scipy.spatial import cKDTree
        #
        print("Building non-periodic tree with SciPy backend...")
        tree = cKDTree(posTracer)
    #
    # Calculate the mean tracer density & mean weights
    mean_tracer_dens = len(posTracer) / (Lbox**3)
    mean_tracer_weight = np.mean(tracerWeights) if tracerWeights is not None else 1.0
    #
    # Arguments to be passed to each worker process
    worker_args = {
        'posVoid': posVoid, 'radVoid': radVoid, 'posTracer': posTracer,
        'velTracer': velTracer, 'tracerWeights': tracerWeights, 'tree': tree,
        'nz': mean_tracer_dens, 'Lbox': Lbox, 'rmax': rmax, 'Nbin': N_radial_bins,
        'periodic': periodic, 'meanWeight': mean_tracer_weight
    }
    #
    r_steps = np.linspace(0, rmax, N_radial_bins + 1)
    #
    with h5py.File(save_path, 'w') as f:
        f.attrs['rmax'] = rmax
        f.attrs['N_radial_bins'] = N_radial_bins
        f.attrs['Lbox'] = Lbox
        f.attrs['backend'] = 'VIDE' if VIDE_AVAILABLE else 'SciPy'
        #
        f.create_dataset('void_ids', data=voidIDs)
        f.create_dataset('void_radii', data=radVoid)
        f.create_dataset('radial_steps', data=r_steps)
        #
        for p_type in profile_types:
            print(f"Calculating individual profiles for: {p_type}...")
            tasks = [(i, p_type) for i in range(len(posVoid))]
            #
            with mp.Pool(processes=N_cpus, initializer=_init_worker_pool, initargs=(worker_args,)) as pool:
                profiles = pool.map(_parallel_task_wrapper, tasks)
                #
            #
            f.create_dataset(p_type, data=np.array(profiles))
            print(f"Finished and saved '{p_type}' profiles.")
        #
    print(f"\nAll individual profiles successfully saved to: {save_path}")



# Processing of pre-calculated individual void profiles
def load_individual_profiles(file_path):
    """
    Loads individual void profiles and metadata from a saved HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file created by
                         `calculate_and_save_individual_profiles`.

    Returns:
        tuple: A tuple containing two dictionaries:
        - data_dict (dict): A dictionary where keys are the names of the
          datasets in the file (e.g., 'void_ids', 'void_radii',
          'number_density', 'velocity') and values are the
          corresponding NumPy arrays.
        - metadata_dict (dict): A dictionary containing the metadata
          stored as attributes in the file (e.g., 'Lbox', 'rmax',
          'N_radial_bins', 'backend').
    """
    #
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")
    #
    data, metadata = {}, {}
    with h5py.File(file_path, 'r') as f:
        for key in f.attrs: metadata[key] = f.attrs[key]
        for key in f.keys(): data[key] = f[key][:]
    #
    print(f"Successfully loaded data for {len(data['void_ids'])} voids.")
    return data, metadata



def select_voids(data, property_name, min_val, max_val, external_property_data=None):
    #
    """
    Selects a subset of voids based on a cut on one of their properties.

    Can filter based on a property within the 'data' dictionary or an
    externally provided array.

    Args:
        data (dict): Data dictionary from `load_individual_profiles`.
        property_name (str): The name of the property to cut on (e.g., 'void_radii').
                             Used as a key if `external_property_data` is None,
                             otherwise used as a label for messages.
        min_val (float): The minimum value for the property (inclusive).
        max_val (float): The maximum value for the property (exclusive).
        external_property_data (np.ndarray, optional): An external NumPy array
            to use for filtering. Must be the same length and order as the
            data in the `data` dictionary. Defaults to None.

    Returns:
        dict: A new dictionary containing only the data for the selected voids.
    """
    #
    if external_property_data is not None:
        # Use the external array for filtering
        # Including a sanity check to ensure the lengths match
        if len(external_property_data) != len(data['void_ids']):
            raise ValueError(
                f"Length of external_property_data ({len(external_property_data)}) does not match "
                f"the number of voids in data ({len(data['void_ids'])})."
            )
            #
        #
        property_array = external_property_data
    else:
        # Use a property from within the data dictionary (original behavior)
        if property_name not in data:
            raise KeyError(f"Property '{property_name}' not found in data.")
        #
        property_array = data[property_name]
    #
    # Create mask to select voids that fit the selection criteria
    mask = (property_array >= min_val) & (property_array < max_val)
    num_selected = np.sum(mask)
    #
    selected_data = {}
    for key, value in data.items():
        #
        if isinstance(value, np.ndarray) and value.shape[0] == len(mask):
            selected_data[key] = value[mask]
        #
        # Also copy 'radial_steps' regardless of its length
        elif key == 'radial_steps':
            selected_data[key] = value
        #
    #
    print(f"Selected {num_selected} voids where {min_val} <= {property_name} < {max_val}.")
    return selected_data



def stack_void_profiles(data, profile_type, velocity_stacking='global',return_density_contrast=True, rmax=None, N_radial_bins=None, nz=None, meanWeight=None):
    """
    Stacks a given set of individual void profiles into a single bin.

    This function takes a dictionary containing arrays of individual void profiles
    (as produced by `load_individual_profiles` or `select_voids`) and
    computes the mean stacked profile and its associated statistical error.

    For density-like profiles ('number_density', 'volume_weighted', etc.), it
    calculates the simple mean and applies the necessary normalization. For velocity
    profiles, it supports two distinct stacking methods to account for the
    nature of the data as a ratio. The covariance matrix and final errors are
    estimated using the robust jackknife resampling technique.
    
    See https://arxiv.org/abs/2210.02457 for more details.

    Args:
        data (dict): A dictionary containing the pre-calculated individual
            profiles and void properties. Must include 'void_ids', 'void_radii',
            and a dataset corresponding to `profile_type`. For velocity profiles,
            it must also include a 'volume_weighted' dataset.
        profile_type (str): The name of the profile dataset within `data` to
            be stacked. Supported types are 'number_density', 'volume_weighted',
            'mass_weighted', and 'velocity'.
        velocity_stacking (str, optional): Specifies the method for stacking
            velocity profiles, which are ratio quantities. Defaults to 'global'.
                - 'global': Sums all numerators and all denominators across
                   the entire void sample before performing a single division.
                   This method is generally more stable in sparse regions.
                - 'individual': First calculates the velocity profile for each
                  void individually, then averages these final profiles.
        return_density_contrast (bool, optional): If True, density-like profiles
            are returned as the density contrast (rho/rho_mean - 1). If False,
            they are returned as the density ratio (rho/rho_mean). Defaults to True.
        rmax (int, optional): The maximum radius of the profiles in units of R_v.
            This is a fallback used only if the radial binning information is not
            found in the input `data` dictionary. Defaults to None.
        N_radial_bins (int, optional): The number of radial bins in the profiles.
            Like `rmax`, this is a fallback used only if binning information is
            not found in `data`. Defaults to None.
        nz (float, optional): The mean number density of tracers. Required when
            stacking 'volume_weighted' or 'mass_weighted' profiles.
        meanWeight (float, optional): The mean weight of tracers. Required when
            stacking 'volume_weighted' or 'mass_weighted' profiles.

    Returns:
        dict: A dictionary containing the results of the stacking analysis:
            - 'N_voids' (int): The number of individual voids in the stack.
            - 'mean_radius' (float): The mean radius of the voids in the stack.
            - 'radial_bins_centers' (np.ndarray): The center points of the radial bins.
            - 'stacked_profile' (np.ndarray): The final mean stacked profile.
            - 'stacked_errors' (np.ndarray): The jackknife error (standard deviation)
              for each point in the stacked profile.
    """
    num_voids = len(data['void_ids'])
    if num_voids < 2:
        print("Cannot stack fewer than 2 profiles.")
        return {}
    #
    # Determine radial bins for normalization
    r_steps_edges = data.get('radial_steps')
    if r_steps_edges is None:
        #
        if rmax is None or N_radial_bins is None:
            raise ValueError("Cannot determine radial bins. Provide 'rmax' and 'N_radial_bins'.")
        #
        r_steps_edges = np.linspace(0, rmax, N_radial_bins + 1)
    #
    mean_radius = np.mean(data['void_radii'])
    shell_volumes_phys = 4.0/3.0 * np.pi * (r_steps_edges[1:]**3 - r_steps_edges[:-1]**3) * mean_radius**3
    #
    #
    if profile_type in ['number_density', 'volume_weighted', 'mass_weighted']:
        main_profiles = data[profile_type]
        #
        main_profiles = data[profile_type]
        mean_profile_raw = np.sum(main_profiles, axis=0) / num_voids
        #
        if profile_type == 'number_density':
            # 'number_density' is already normalized in _calc_one_profile, so we just average
            mean_profile = mean_profile_raw
        else: # 'volume_weighted' or 'mass_weighted'
            #
            if nz is None or meanWeight is None:
                raise ValueError("Must provide 'nz' and 'meanWeight' for weighted density profile stacking.")
            #
            weighted_density = mean_profile_raw / shell_volumes_phys
            mean_weighted_density = nz * meanWeight
            mean_profile = weighted_density / mean_weighted_density
        #
        if return_density_contrast:
            mean_profile -= 1.0
        #
        cov_matrix = _jackknife(main_profiles, mean_profile)
        #
    elif profile_type == 'velocity':
        #
        main_profiles = data[profile_type]
        denom_key = 'volume_weighted'
        if denom_key not in data:
            raise KeyError(f"Required denominator '{denom_key}' not found in data.")
            #
        #
        divisor_profiles = data[denom_key]
        #
        if velocity_stacking == 'global': # Global Stack
            sum_numerators = np.sum(main_profiles, axis=0)
            sum_denominators = np.sum(divisor_profiles, axis=0)
            #
            # Safety check before division
            safe_denominators = np.where(sum_denominators == 0, 1, sum_denominators)
            mean_profile = sum_numerators / safe_denominators
            mean_profile[sum_denominators == 0] = 0 # Set result to 0 where the original sum was 0
            #
            # Calculate the covariance matrix
            jackknife_numerators = sum_numerators - main_profiles
            jackknife_denominators = sum_denominators - divisor_profiles
            # Avoid division by zero when calculating the leave-one-out ratios
            safe_jackknife_denominators = np.where(jackknife_denominators == 0, 1, jackknife_denominators)
            jackknife_ratios = jackknife_numerators / safe_jackknife_denominators
            jackknife_ratios[jackknife_denominators == 0] = 0 # Correct the result
            #
            jackknife_samples = jackknife_ratios - mean_profile
            cov_matrix = (num_voids - 1.)**2 / num_voids * np.cov(jackknife_samples, rowvar=0)
            #
        elif velocity_stacking == 'individual': # Individual Stack
            #
            ratios = main_profiles / np.where(divisor_profiles == 0, 1, divisor_profiles)
            ratios[divisor_profiles == 0] = 0
            #
            sum_of_ratios = np.sum(ratios, axis=0)
            mean_profile = sum_of_ratios / num_voids
            cov_matrix = _jackknife(ratios, mean_profile)
            #
        else:
            raise ValueError(f"Invalid 'velocity_stacking' option. Choose 'global' or 'individual'.")
        #
    else:
        raise ValueError(f"Profile type '{profile_type}' not supported.")
    #
    # Creating the bin centers
    radial_bins_centers = None
    if 'radial_steps' in data:
        # Best case: Infer directly from the saved data
        r_steps_edges = data['radial_steps']
        radial_bins_centers = (r_steps_edges[:-1] + r_steps_edges[1:]) / 2.0
    elif rmax is not None and N_radial_bins is not None:
        # Fallback: Use user-provided arguments
        print("Warning: 'radial_steps' not in data dict. Using provided rmax and N_radial_bins.")
        r_steps_edges = np.linspace(0, rmax, N_radial_bins + 1)
        radial_bins_centers = (r_steps_edges[:-1] + r_steps_edges[1:]) / 2.0
    else:
        print("Warning: Cannot determine radial bins. Provide 'rmax' and 'N_radial_bins', or ensure 'radial_steps' is in the data dictionary.")
    #
    return {
        'N_voids': num_voids,
        'mean_radius': np.mean(data['void_radii']),
        'radial_bins_centers': radial_bins_centers,
        'stacked_profile': mean_profile,
        'stacked_errors': np.sqrt(np.diag(cov_matrix))
    }




