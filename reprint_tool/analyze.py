import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

def normalize(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

def calculate_rmse(x, y):
    x_normalized = normalize(x)
    y_normalized = normalize(y)
    return np.sqrt(np.nanmean((x_normalized - y_normalized) ** 2))

def calculate_cosine(x, y):
    return 1 - np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def create_heatmap_with_custom_sim(df, calc_func=calculate_rmse, colorscale='Blues', hide_heatmap=False, method='complete'):
    """
    Create a heatmap with dendrograms using matplotlib.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with signatures as columns
    calc_func : function
        Function to calculate similarity/distance (default: calculate_rmse)
    colorscale : str
        Matplotlib colormap name (default: 'Blues')
    hide_heatmap : bool
        If True, show only dendrograms (default: False)
    method : str
        Linkage method for clustering (default: 'complete')
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with heatmap and dendrograms
    """
    # Transpose data and get labels
    df = df.T
    labels = df.index.tolist()

    n = df.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = calc_func(df.iloc[i, :], df.iloc[j, :])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method=method)
    
    # Get dendrogram leaf order
    # Create a temporary dendrogram to get the leaf order
    fig_temp = plt.figure(figsize=(1, 1))
    ax_temp = fig_temp.add_subplot(111)
    ddata = dendrogram(Z, no_plot=True)
    plt.close(fig_temp)
    
    dendro_leaves = ddata['leaves']
    reordered_labels = [labels[i] for i in dendro_leaves]
    
    # Reorder distance matrix according to dendrogram
    if not hide_heatmap:
        heat_data = dist_matrix[dendro_leaves, :]
        heat_data = heat_data[:, dendro_leaves]
    
    # Create figure with gridspec for complex layout
    fig = plt.figure(figsize=(10, 10))
    
    if hide_heatmap:
        # Only dendrograms - simpler layout
        gs = gridspec.GridSpec(2, 2, width_ratios=[0.15, 1], height_ratios=[0.15, 1], 
                              hspace=0.05, wspace=0.05)
        
        # Top dendrogram (horizontal)
        ax_top = fig.add_subplot(gs[0, 1])
        dendrogram(Z, ax=ax_top, orientation='top', labels=reordered_labels, 
                  leaf_rotation=90, leaf_font_size=8)
        ax_top.axis('off')
        
        # Left dendrogram (vertical)
        ax_left = fig.add_subplot(gs[1, 0])
        dendrogram(Z, ax=ax_left, orientation='left', labels=reordered_labels,
                  leaf_font_size=8)
        ax_left.axis('off')
        
        # Empty space for alignment
        ax_empty = fig.add_subplot(gs[1, 1])
        ax_empty.axis('off')
        
    else:
        # Full layout with heatmap
        # Use 4 columns: [dendrogram_left, heatmap, labels_y_right, colorbar_space] and 2 rows: [dendrogram_top, main_content]
        # Smaller gap between dendrogram and heatmap (wspace=0.01)
        gs = gridspec.GridSpec(2, 4, width_ratios=[0.10, 1, 0.12, 0.05], height_ratios=[0.15, 1], 
                              hspace=0.05, wspace=0.01)
        
        # Top dendrogram (horizontal) - spans heatmap column only
        ax_top = fig.add_subplot(gs[0, 1])
        dendrogram(Z, ax=ax_top, orientation='top', no_labels=True, color_threshold=0)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.spines['left'].set_visible(False)
        
        # Left dendrogram (vertical) - closer to heatmap
        ax_left = fig.add_subplot(gs[1, 0])
        dendrogram(Z, ax=ax_left, orientation='left', no_labels=True, color_threshold=0)
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_left.spines['top'].set_visible(False)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['bottom'].set_visible(False)
        ax_left.spines['left'].set_visible(False)
        
        # Shorten labels by removing '_reprint' suffix
        display_labels = []
        for label in reordered_labels:
            if label.endswith('_reprint'):
                display_labels.append(label[:-8])  # Remove '_reprint' (8 characters)
            else:
                display_labels.append(label)
        
        # Main heatmap
        ax_heatmap = fig.add_subplot(gs[1, 1])
        
        # Get colormap
        try:
            cmap = cm.get_cmap(colorscale)
        except:
            cmap = cm.get_cmap('Blues')  # Fallback
        
        # Reverse colormap if needed (similar to reversescale=True in plotly)
        im = ax_heatmap.imshow(heat_data, aspect='auto', origin='lower', 
                               cmap=cmap, interpolation='nearest')
        
        # Set ticks and labels for X-axis only (Y labels are on the right)
        ax_heatmap.set_xticks(np.arange(len(reordered_labels)))
        ax_heatmap.set_xticklabels(display_labels, rotation=90, ha='right', va='top', fontsize=10)
        ax_heatmap.set_yticks([])  # No Y ticks on heatmap
        ax_heatmap.set_yticklabels([])  # No Y labels on heatmap
        
        # Get Y limits from heatmap for alignment
        ylim_heatmap = ax_heatmap.get_ylim()
        
        # Y-axis labels - on the RIGHT side of heatmap
        ax_labels_y = fig.add_subplot(gs[1, 2])
        ax_labels_y.set_xlim(0, 1)
        ax_labels_y.set_ylim(ylim_heatmap)  # Match heatmap Y limits
        
        # Set Y labels on the right side - align with heatmap rows
        y_positions = np.arange(len(reordered_labels))
        ax_labels_y.set_yticks(y_positions)
        ax_labels_y.set_yticklabels(display_labels, fontsize=11, ha='left', va='center')
        
        # Hide all axis elements except labels
        ax_labels_y.set_xticks([])
        ax_labels_y.set_xticklabels([])
        ax_labels_y.tick_params(axis='y', which='both', length=0, width=0, pad=0, labelsize=11)  # Increased padding
        ax_labels_y.tick_params(axis='x', which='both', length=0, width=0)
        
        # Remove all spines
        ax_labels_y.spines['top'].set_visible(False)
        ax_labels_y.spines['right'].set_visible(False)
        ax_labels_y.spines['bottom'].set_visible(False)
        ax_labels_y.spines['left'].set_visible(False)
        
        # Remove grid
        ax_labels_y.grid(False)
        
        # Remove all axis elements for cleaner look - no axis lines, only labels
        ax_heatmap.set_xlabel('')
        ax_heatmap.set_ylabel('')
        
        # Hide tick marks but keep labels for X-axis
        ax_heatmap.tick_params(axis='x', which='both', 
                              length=0, width=0, 
                              pad=8,  # Increased padding for better spacing
                              labelsize=10)
        ax_heatmap.tick_params(axis='y', which='both', 
                              length=0, width=0)
        
        # Remove all spines and axis lines completely
        for spine in ax_heatmap.spines.values():
            spine.set_visible(False)
        
        # Remove grid
        ax_heatmap.grid(False)
        
        # Hide axis lines
        ax_heatmap.axhline(y=0, color='none', linewidth=0)
        ax_heatmap.axvline(x=0, color='none', linewidth=0)
        
        # Add colorbar - positioned after labels
        # Create space for colorbar on the right (adjusted for new layout with visible labels)
        cbar_ax = fig.add_axes([0.87, 0.20, 0.02, 0.62])  # [left, bottom, width, height]
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=9)
        
        # Empty top-left corner
        ax_empty = fig.add_subplot(gs[0, 0])
        ax_empty.axis('off')
        
        # Empty space for colorbar column
        ax_cbar_space = fig.add_subplot(gs[1, 3])
        ax_cbar_space.axis('off')
    
    # Use subplots_adjust instead of tight_layout for better control
    # Adjust margins to accommodate labels on the right side - more space for right labels
    if not hide_heatmap:
        plt.subplots_adjust(left=0.12, right=0.85, top=0.82, bottom=0.22)
    else:
        plt.subplots_adjust(left=0.12, right=0.95, top=0.82, bottom=0.22)
    
    return fig
