Void Profile Analysis Toolkit:

A Python toolkit for calculating, saving, and stacking radial profiles around cosmic voids from cosmological simulation data. The core functionality is to measure physical properties—such as tracer density and velocity—in spherical shells as a function of distance from a void's center. The toolkit includes robust methods for error estimation using Jackknife resampling. Based on https://arxiv.org/abs/2210.02457 and https://arxiv.org/abs/2312.11241




Table of Contents:

Overview

Features

Installation

Workflow & Usage

Step 1: Prepare Your Data

Step 2: Calculate Individual Profiles

Step 3: Load Pre-calculated Profiles

Step 4: Select Voids and Stack Profiles

Step 5: Visualize the Results

Methodology

Overview In cosmology, the large-scale structure of the universe is a cosmic web of filaments, clusters, and voids. Voids are the vast, underdense regions that contain very few galaxies. Studying the properties of matter and galaxies around voids provides crucial insights into the nature of dark energy, gravity, and galaxy formation.

This toolkit provides a complete workflow for void stacking, a technique where we average the radial profiles of many individual voids to measure a high signal-to-noise profile. The process involves:

Calculating individual profiles for voids in parallel.

Saving these intermediate results to an HDF5 file, so they can be analyzed later without recalculating.

Loading the data and selecting voids based on specific properties (e.g., radius).

Stacking the selected profiles to get a mean profile and estimating the statistical error using the Jackknife method.

Features: High-Performance: Uses the multiprocessing module to calculate profiles in parallel, significantly speeding up the analysis of large datasets.

Flexible Backend: Automatically uses the optimized VIDE library for periodic k-d tree searches if installed. Falls back to a scipy.spatial.cKDTree implementation (replicating the box) if VIDE is not found.

Multiple Profile Types: Natively supports density and velocity profiles:

    number_density: The number of tracers per unit volume.

    volume_weighted / mass_weighted: Density profiles weighted by a tracer property (e.g., volume or mass).

    velocity: The average radial velocity profile, crucial for studying void dynamics.

Robust Error Estimation: Implements the Jackknife resampling technique to calculate the covariance matrix and the standard error on the stacked profile.

Modular Workflow: You can calculate profiles once and save them, then experiment with different selection criteria and stacking options later.

Installation

This toolkit requires Python 3 and the following libraries:

numpy, h5py, scipy (used as a fallback)

For the best performance with periodic data, it is highly recommended to also install the VIDE toolkit:

https://bitbucket.org/cosmicvoids/vide_public/src/master/

The code will automatically detect if VIDE is available and use it.

Workflow & Usage A complete example, from generating dummy data to plotting the final stacked profile is included in example_analysis.py

Here is a brief overview of the main functions. See the docstrings in the source code for full details on all parameters.

calculate_and_save_individual_profiles(...)

The main workhorse function that computes profiles for all voids in parallel and saves them.

Key args: save_path, posVoid, radVoid, posTracer, Lbox, profile_types, N_cpus.

load_individual_profiles(file_path)

Loads the data and metadata from an HDF5 file created by the function above.

Returns: A data dictionary and a metadata dictionary.

select_voids(data, property_name, min_val, max_val)

Filters the loaded data to select a subset of voids based on a property (e.g., void_radii).

Returns: A new data dictionary containing only the selected voids.

stack_void_profiles(data, profile_type, ...)

Takes a data dictionary (either the full set or a selection) and computes the mean stacked profile and Jackknife errors.

Key args: data, profile_type, velocity_stacking, return_density_contrast.

Returns: A dictionary containing the final results (stacked_profile, stacked_errors, etc.).

Methodology

The calculations performed by this code are based on standard techniques in large-scale structure analysis. For a detailed reference on the methodology, especially concerning the stacking of ratio quantities like velocity profiles and the use of Jackknife errors, please see arXiv:2210.02457.
