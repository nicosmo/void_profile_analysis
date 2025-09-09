
# This script demonstrates a complete workflow using the void_profiles_library:
# 1. Generates mock data for voids and tracers.
# 2. Calculates and saves individual void profiles to an HDF5 file.
# 3. Loads the saved profiles back into memory.
# 4. Selects a subset of voids based on their radius.
# 5. Stacks the selected profiles to get a final average profile with errors.
# 6. Plots the final stacked profiles.

import numpy as np
import matplotlib.pyplot as plt

# Import the functions from your custom library file
import void_profiles_library as voidProfiles




# --- 1. Generate Mock Data ---
print("\n--- Generating Mock Data ---")
LBOX = 1000.0  # Box size in Mpc/h
N_VOIDS = 10000
N_TRACERS = 5000000

np.random.seed(42)
mock_posVoid = np.random.rand(N_VOIDS, 3) * LBOX
mock_radVoid = np.random.uniform(10, 50, N_VOIDS)
mock_voidIDs = np.arange(N_VOIDS)
mock_posTracer = np.random.rand(N_TRACERS, 3) * LBOX
mock_velTracer = np.random.normal(0, 100, (N_TRACERS, 3))
mock_tracerWeights = np.random.uniform(0.5, 1.5, N_TRACERS)  # Mock volumes for weighting



# --- 2. Calculate and Save Individual Profiles ---
SAVE_FILE = "mock_individual_profiles.hdf5"
PROFILES_TO_CALC = ['number_density', 'velocity', 'volume_weighted']

print(f"\n--- Calculating and Saving {len(PROFILES_TO_CALC)} Profile Types ---")
voidProfiles.calculate_and_save_individual_profiles(
    save_path=SAVE_FILE,
    voidIDs=mock_voidIDs,
    posVoid=mock_posVoid,
    radVoid=mock_radVoid,
    posTracer=mock_posTracer,
    velTracer=mock_velTracer,
    tracerWeights=mock_tracerWeights,
    Lbox=LBOX,
    profile_types=PROFILES_TO_CALC,
    rmax=5,
    N_radial_bins=50,
    N_cpus=8
)



# --- 3. Load the Saved Profiles ---
print("\n--- Loading Data ---")
loaded_data, loaded_metadata = voidProfiles.load_individual_profiles(SAVE_FILE)



# --- 4. Select a Subset of Voids ---
print("\n--- Selecting a Subset of Voids ---")
# Select voids with radii between 20 and 40 Mpc/h for this example
selected_voids_data = voidProfiles.select_voids(
    data=loaded_data,
    property_name='void_radii',
    min_val=20.0,
    max_val=40.0
)



# --- 5. Stack the Selected Profiles ---
print("\n--- Stacking Selected Profiles ---")

# Pre-calculate mean density and weight needed for the stacking function
mean_tracer_dens = len(mock_posTracer) / (LBOX**3)
mean_tracer_weight = np.mean(mock_tracerWeights)

# First, stack the number density profile
stacked_number_density = voidProfiles.stack_void_profiles(
    data=selected_voids_data,
    profile_type='number_density',
    return_density_contrast=True, # Returns rho/rho_bar - 1
    nz=mean_tracer_dens,
    meanWeight=mean_tracer_weight
)

# Second, stack the velocity profile using the "global stack" method
stacked_velocity_global = voidProfiles.stack_void_profiles(
    data=selected_voids_data,
    profile_type='velocity',
    velocity_stacking='global',
    nz=mean_tracer_dens,
    meanWeight=mean_tracer_weight
)

# Third, stack the velocity profile using the "individual stack" method
stacked_velocity_individual = voidProfiles.stack_void_profiles(
    data=selected_voids_data,
    profile_type='velocity',
    velocity_stacking='individual',
    nz=mean_tracer_dens,
    meanWeight=mean_tracer_weight
)


print("\n--- Stacking Complete ---")
if stacked_number_density:
    #
    print(f"Stacked {stacked_number_density['N_voids']} voids for profiles.")
    print(f"Mean radius of stacked voids: {stacked_number_density['mean_radius']:.2f} Mpc/h")
    #
#



# --- 6. Plot the Results ---
try:
    #
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Stacked Void Profiles for Mock Data ($20 < R_v < 40$ Mpc/h)", fontsize=16)
    #
    # Plot number_density
    if stacked_number_density:
        #
        ax1.errorbar(
            stacked_number_density['radial_bins_centers'],
            stacked_number_density['stacked_profile'],
            yerr=stacked_number_density['stacked_errors'],
            fmt='o-',
            capsize=3,
            label='$n_{\\mathrm{norm}}$'
        )
        #
    #
    ax1.axhline(0, color='k', linestyle='--', lw=1)
    ax1.set_ylabel("$\\rho/\\bar{\\rho} - 1$")
    ax1.legend()
    ax1.grid(alpha=0.3)
    #
    # Plot global velocity stack
    if stacked_velocity_global:
        #
        ax2.errorbar(
            stacked_velocity_global['radial_bins_centers'],
            stacked_velocity_global['stacked_profile'],
            yerr=stacked_velocity_global['stacked_errors'],
            fmt='s-',
            capsize=3,
            color='r',
            label='$v$ (Global Stack)'
        )
        #
    #
    # Plot individual velocity stack
    if stacked_velocity_individual:
        #
        ax2.errorbar(
            stacked_velocity_individual['radial_bins_centers'],
            stacked_velocity_individual['stacked_profile'],
            yerr=stacked_velocity_individual['stacked_errors'],
            fmt='^--',
            capsize=3,
            color='b',
            label='$v$ (Individual Stack)'
        )
        #
    #
    ax2.axhline(0, color='k', linestyle='--', lw=1)
    ax2.set_ylabel("$u_v(r)$ [km/s]")
    ax2.set_xlabel("$r / r_v$")
    ax2.legend()
    ax2.grid(alpha=0.3)
    #
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = "stacked_mock_profile_comparison.pdf"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    #
except ImportError:
    #
    print("\nMatplotlib not found. Skipping plot.")
    #
except Exception as e:
    #
    print(f"\nAn error occurred during plotting: {e}")





