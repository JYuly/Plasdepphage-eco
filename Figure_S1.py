import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# File paths - modify these to match your data file locations
data_files = [
    'phase_diagram_data_low_infectivity.npz',  # or .pkl for pickle files
    'phase_diagram_data_med_infectivity.npz',
    'phase_diagram_data_high_infectivity.npz'
]

# Load data from three files
datasets = []
for file_path in data_files:
    # Load .npz files which contain multiple arrays
    data = dict(np.load(file_path, allow_pickle=True))
    datasets.append(data)

# Create figure with 2 rows and 3 columns (use constrained_layout)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

# Store mesh objects for colorbar creation
plas_meshes = []
pred_meshes = []
levels1 = np.logspace(-4, 0, 13)
levels2 = np.logspace(-3, 0, 13)

# Plot each dataset
kappa_values = [0.1, 1.0, 10.0]
for col_idx, dataset in enumerate(datasets):
    #######################
    # First row: plas_data
    #######################

    cmap1 = matplotlib.colormaps['BuGn'].copy()
    cmap1.set_bad(color='gray') #Make color grey when species are extinct (density = 0 cannot be plotted on a log plot)
    norm1 = matplotlib.colors.LogNorm(vmin=levels1[0], vmax=levels1[-1])
    
    # Use provided X/Y meshgrid if available, otherwise fall back to array indices
    X = dataset.get('X') if isinstance(dataset, dict) else None
    Y = dataset.get('Y') if isinstance(dataset, dict) else None
    if X is not None and Y is not None:
        plas_mesh = axes[0, col_idx].pcolormesh(X, Y, dataset['plas_ave'], shading='auto', cmap=cmap1, norm=norm1)
    else:
        plas_mesh = axes[0, col_idx].pcolormesh(dataset['plas_ave'], shading='auto', cmap=cmap1, norm=norm1)
    plas_meshes.append(plas_mesh)
    axes[0, col_idx].set_xscale('log')
    axes[0, col_idx].set_yscale('log')
    axes[0, col_idx].set_title('Plasmid-carrying fraction',fontsize = 16)
    axes[0, col_idx].text(
        0.02, 0.95,
        rf'$\kappa = {kappa_values[col_idx]:.1f}$',
        transform=axes[0, col_idx].transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
    )
    # y-axis label will be applied at the figure level

    #########################
    # Second row: pred_data
    #########################

    cmap2 = matplotlib.colormaps['Reds'].copy()
    cmap2.set_bad(color='gray') #Make color grey when species are extinct (density = 0 cannot be plotted on a log plot)
    norm2 = matplotlib.colors.LogNorm(vmin=levels2[0], vmax=levels2[-1])

    if X is not None and Y is not None:
        pred_mesh = axes[1, col_idx].pcolormesh(X, Y, dataset['pred_ave'], shading='auto', cmap=cmap2, norm=norm2)
    else:
        pred_mesh = axes[1, col_idx].pcolormesh(dataset['pred_ave'], shading='auto', cmap=cmap2, norm=norm2)
    pred_meshes.append(pred_mesh)
    axes[1, col_idx].set_xscale('log')
    axes[1, col_idx].set_yscale('log')
    axes[1, col_idx].set_title('Relative predation rate ρ', fontsize = 14)
    axes[1, col_idx].text(
        0.02, 0.95,
        rf'$\kappa = {kappa_values[col_idx]:.1f}$',
        transform=axes[1, col_idx].transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
    )
    # x-axis label will be applied at the figure level

# Set tick label sizes for all subplots
for ax in axes.flat:
    ax.tick_params(labelsize=14)

# Use constrained_layout and let Matplotlib place shared colorbars without overlapping
# Shared colorbar for top row
cbar1 = fig.colorbar(plas_meshes[-1], ax=axes[0, :], pad=0.03, fraction=0.03)
ticks1 = np.logspace(np.log10(levels1[0]), np.log10(levels1[-1]), num=5)
cbar1.set_ticks(ticks1)
ticklabels1 = [lab.get_text() for lab in cbar1.ax.get_yticklabels()]
ticklabels1[0] = '≤' + ticklabels1[0]
cbar1.ax.set_yticklabels(ticklabels1, fontsize=14)

# Shared colorbar for bottom row
cbar2 = fig.colorbar(pred_meshes[-1], ax=axes[1, :], pad=0.03, fraction=0.03)
ticks2 = np.logspace(-3, 0, num=4)
cbar2.set_ticks(ticks2)
ticklabels2 = [lab.get_text() for lab in cbar2.ax.get_yticklabels()]
ticklabels2[0] = '≤' + ticklabels2[0]
cbar2.ax.set_yticklabels(ticklabels2, fontsize=14)

# Add shared axis labels below creating the figure
fig.supxlabel('Repressed conjugation rate $\gamma_{\mathrm{lo}}$ (ml hr$^{-1}$)', fontsize=14)
fig.supylabel('Repression rate $1/τ$ (hr$^{-1}$)', fontsize=14)

plt.savefig('Figure_S1.png', format='png', dpi=400, bbox_inches='tight')
