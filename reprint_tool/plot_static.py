#!/usr/bin/env python3
"""
Generate seamless 3-panel stacked bar chart visualizations for RePrints.
Features: no y-axes, centered middle panel, compact 16x4.5 inch format.
Creates individual PNG files and a combined PDF with all RePrint plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import sys

# COSMIC-like colors for mutations
COLORS_C = ['#00BFFF', '#000000', '#E32636']  # C>A (cyan), C>G (black), C>T (red)
COLORS_T = ['#C0C0C0', '#90EE90', '#FFB6C1']  # T>A (gray), T>C (green), T>G (pink)

# Define trinucleotide contexts
BASES = ['A', 'C', 'G', 'T']
C_CONTEXT_LABELS = [f"{l}C{r}" for l in BASES for r in BASES]
T_CONTEXT_LABELS = [f"{l}T{r}" for l in BASES for r in BASES]
ALL_CONTEXT_LABELS = C_CONTEXT_LABELS + T_CONTEXT_LABELS


def load_reprint_data(csv_path):
    """Load RePrint data from CSV file."""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def extract_reprint_vector(df, reprint_name):
    """Extract a single RePrint vector from the dataframe, grouped by trinucleotide context."""
    if reprint_name not in df.columns:
        raise ValueError(f"RePrint '{reprint_name}' not found in data")
    
    reprint_data = df[reprint_name]
    
    # Initialize arrays for 32 contexts x 3 mutations each
    c_probs = np.zeros((16, 3))  # 16 C-centered contexts
    t_probs = np.zeros((16, 3))  # 16 T-centered contexts
    
    # Map each row index to the correct position in our arrays
    for idx, value in reprint_data.items():
        # Parse the mutation category: L[X>Y]R
        parts = idx.split('[')
        left_base = parts[0]
        mutation_and_right = parts[1].split(']')
        mutation = mutation_and_right[0]
        right_base = mutation_and_right[1]
        
        ref_base = mutation.split('>')[0]
        mut_base = mutation.split('>')[1]
        
        # Get context index (0-15 for each of 16 possible LXR combinations)
        bases_list = ['A', 'C', 'G', 'T']
        left_idx = bases_list.index(left_base)
        right_idx = bases_list.index(right_base)
        context_idx = left_idx * 4 + right_idx
        
        # Get mutation index (0-2 for the 3 possible mutations)
        if ref_base == 'C':
            mut_order = ['A', 'G', 'T']
            mut_idx = mut_order.index(mut_base)
            c_probs[context_idx, mut_idx] = value
        else:  # ref_base == 'T'
            mut_order = ['A', 'C', 'G']
            mut_idx = mut_order.index(mut_base)
            t_probs[context_idx, mut_idx] = value
    
    return c_probs, t_probs


def plot_reprint_seamless(reprint_name, c_probs, t_probs):
    """
    Generate a seamless 3-panel bar chart visualization for a single RePrint.
    
    Features:
    - All three bars in each column sum to height 1.0
    - Top panel: bars aligned at top (inverted y-axis)
    - Middle panel: bars centered vertically
    - Bottom panel: bars aligned at bottom
    - No y-axes, seamless appearance
    - Compact format matching signature plots
    
    Parameters:
    -----------
    reprint_name : str
        Name of the RePrint signature
    c_probs : np.array
        16x3 array of probabilities for C-centered contexts
    t_probs : np.array
        16x3 array of probabilities for T-centered contexts
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    
    # Adjusted size to better match signature plots (closer to 12x8 aspect ratio)
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(14, 5.5), 
                                                   sharex=True, 
                                                   gridspec_kw={'hspace': 0.0})
    
    # Create x positions with a gap between C and T contexts
    x_pos_c = np.arange(16)
    x_pos_t = np.arange(16) + 16.5
    x_pos = np.concatenate([x_pos_c, x_pos_t])
    width = 0.8
    
    # All panels use the same scale: 0 to 1.0
    # Each column's three bars will sum to 1.0 in total height
    
    # BOTTOM PANEL: C>A (cyan) and T>A (gray) - from 0 to their values
    ax_bot.bar(x_pos_c, c_probs[:, 0], width, color=COLORS_C[0], edgecolor='white', linewidth=0.5)
    ax_bot.bar(x_pos_t, t_probs[:, 0], width, color=COLORS_T[0], edgecolor='white', linewidth=0.5)
    ax_bot.set_ylim(0, 1.0)
    ax_bot.set_xlim(-0.5, max(x_pos) + 0.5)
    
    # MIDDLE PANEL: C>G (black) and T>C (green) - centered at their midpoints
    c_heights_mid = c_probs[:, 1]
    t_heights_mid = t_probs[:, 1]
    ax_mid.bar(x_pos_c, c_heights_mid, width, color=COLORS_C[1], edgecolor='white', 
               linewidth=0.5, bottom=-c_heights_mid/2)
    ax_mid.bar(x_pos_t, t_heights_mid, width, color=COLORS_T[1], edgecolor='white', 
               linewidth=0.5, bottom=-t_heights_mid/2)
    ax_mid.set_ylim(-0.5, 0.5)  # Centered around 0, total range = 1.0
    ax_mid.set_xlim(-0.5, max(x_pos) + 0.5)
    
    # TOP PANEL: C>T (red) and T>G (pink) - bars from top down
    ax_top.bar(x_pos_c, c_probs[:, 2], width, color=COLORS_C[2], edgecolor='white', linewidth=0.5)
    ax_top.bar(x_pos_t, t_probs[:, 2], width, color=COLORS_T[2], edgecolor='white', linewidth=0.5)
    ax_top.set_ylim(0, 1.0)
    ax_top.invert_yaxis()
    ax_top.set_xlim(-0.5, max(x_pos) + 0.5)
    
    # Remove all axes
    for ax in [ax_top, ax_mid, ax_bot]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
    
    ax_top.set_xticks([])
    ax_mid.set_xticks([])
    
    # Add x-axis labels to bottom panel only
    x_labels_formatted = []
    for label in ALL_CONTEXT_LABELS:
        left = label[0]
        center = label[1]
        right = label[2]
        formatted = f'{left}$\\mathbf{{{center}}}${right}'
        x_labels_formatted.append(formatted)
    
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels(x_labels_formatted, rotation=90, fontsize=15)
    ax_bot.tick_params(axis='x', length=0)
    
    # Add title
    display_name = reprint_name.replace('reprint_', '')
    fig.suptitle(f'RePrint - {display_name}', fontsize=20, fontweight='bold', y=0.98)
    
    # Add legends - positioned closer to plot area
    # Calculate panel positions in figure coordinates
    # Each panel takes roughly 1/3 of the figure height (excluding title)
    panel_height = 0.28  # Approximate height of each panel in figure coordinates
    panel_bottom = 0.08  # Bottom of bottom panel
    
    legend_y_bot = panel_bottom + panel_height * 0.4  # ~0.19
    legend_y_mid = panel_bottom + panel_height * 1.4  # ~0.47
    legend_y_top = panel_bottom + panel_height * 2.4  # ~0.75
    legend_x_left = 0.03  # Moved closer to plot
    legend_x_right = 0.97  # Moved closer to plot
    
    def add_fig_legend(x_pos, y_pos, label, color):
        rect = plt.Rectangle((x_pos, y_pos), 0.01, 0.06, 
                            transform=fig.transFigure, 
                            facecolor=color, edgecolor='none', clip_on=False)
        fig.patches.append(rect)
        fig.text(x_pos + 0.005, y_pos + 0.08, label, 
                transform=fig.transFigure,
                rotation=90, va='bottom', ha='center', 
                fontsize=14, fontweight='bold')
    
    # Add legends for each panel
    add_fig_legend(legend_x_left, legend_y_bot, 'C>A', COLORS_C[0])
    add_fig_legend(legend_x_right, legend_y_bot, 'T>A', COLORS_T[0])
    
    add_fig_legend(legend_x_left, legend_y_mid, 'C>G', COLORS_C[1])
    add_fig_legend(legend_x_right, legend_y_mid, 'T>C', COLORS_T[1])
    
    add_fig_legend(legend_x_left, legend_y_top, 'C>T', COLORS_C[2])
    add_fig_legend(legend_x_right, legend_y_top, 'T>G', COLORS_T[2])
    
    # Add scale indicator showing what 1.0 represents
    # Place scale on the middle panel (centered) as a reference
    scale_x_pos = max(x_pos) + 1.5  # Just after the last bar
    scale_height = 1.0
    
    # Draw scale bar on middle panel (centered around 0)
    ax_mid.plot([scale_x_pos, scale_x_pos], [-scale_height/2, scale_height/2], 
                'k-', linewidth=2, clip_on=False)
    ax_mid.plot([scale_x_pos - 0.1, scale_x_pos + 0.1], [-scale_height/2, -scale_height/2], 
                'k-', linewidth=2, clip_on=False)
    ax_mid.plot([scale_x_pos - 0.1, scale_x_pos + 0.1], [scale_height/2, scale_height/2], 
                'k-', linewidth=2, clip_on=False)
    ax_mid.text(scale_x_pos + 0.3, 0, '1.0', 
                ha='left', va='center', fontsize=12, fontweight='bold',
                transform=ax_mid.transData)
    
    # Set white background
    fig.patch.set_facecolor('white')
    
    return fig


def generate_single_reprint(csv_path, reprint_name, output_path):
    """
    Generate a single RePrint plot and save it.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing RePrint data
    reprint_name : str
        Name of the RePrint to generate (e.g., 'reprint_SBS6')
    output_path : str
        Path where the output PNG will be saved
    """
    df = load_reprint_data(csv_path)
    c_probs, t_probs = extract_reprint_vector(df, reprint_name)
    fig = plot_reprint_seamless(reprint_name, c_probs, t_probs)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_path}")


def generate_all_reprints(csv_path, output_pdf=None, output_dir=None):
    """
    Generate RePrint plots for all signatures in the CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing RePrint data
    output_pdf : str, optional
        Path to output PDF file containing all plots
    output_dir : str, optional
        Directory to save individual PNG files
    """
    print(f"Loading RePrint data from {csv_path}...")
    df = load_reprint_data(csv_path)
    
    # Support both 'reprint_*' and '*_reprint' column naming conventions
    reprint_names = [col for col in df.columns if col.startswith('reprint_') or col.endswith('_reprint')]
    n_reprints = len(reprint_names)
    
    print(f"Found {n_reprints} RePrints to process")
    
    # Generate PDF with all plots
    if output_pdf:
        print(f"Generating combined PDF: {output_pdf}")
        with PdfPages(output_pdf) as pdf:
            for i, reprint_name in enumerate(reprint_names, 1):
                print(f"  Processing {i}/{n_reprints}: {reprint_name}")
                
                try:
                    c_probs, t_probs = extract_reprint_vector(df, reprint_name)
                    fig = plot_reprint_seamless(reprint_name, c_probs, t_probs)
                    
                    # Save to PDF
                    pdf.savefig(fig, dpi=150, bbox_inches='tight', facecolor='white')
                    
                    # Optionally save individual PNG
                    if output_dir:
                        png_path = f"{output_dir}/{reprint_name}_seamless.png"
                        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
                    
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"    Error processing {reprint_name}: {e}")
                    continue
        
        print(f"✓ PDF saved: {output_pdf}")
    
    # Generate individual PNG files only
    elif output_dir:
        print(f"Generating individual PNG files in: {output_dir}")
        for i, reprint_name in enumerate(reprint_names, 1):
            print(f"  Processing {i}/{n_reprints}: {reprint_name}")
            
            try:
                c_probs, t_probs = extract_reprint_vector(df, reprint_name)
                fig = plot_reprint_seamless(reprint_name, c_probs, t_probs)
                
                png_path = f"{output_dir}/{reprint_name}_seamless.png"
                fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
            except Exception as e:
                print(f"    Error processing {reprint_name}: {e}")
                continue
        
        print(f"✓ PNG files saved in: {output_dir}")
    
    print("\nDone!")


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python generate_reprint_seamless.py <reprints.csv> [output.pdf] [output_dir]")
        print("\nExamples:")
        print("  # Generate PDF only:")
        print("  python generate_reprint_seamless.py reprints.csv all_reprints_seamless.pdf")
        print("\n  # Generate PDF and individual PNGs:")
        print("  python generate_reprint_seamless.py reprints.csv all_reprints_seamless.pdf ./png_outputs/")
        print("\n  # Generate individual PNGs only:")
        print("  python generate_reprint_seamless.py reprints.csv - ./png_outputs/")
        print("\n  # Generate single RePrint:")
        print("  from generate_reprint_seamless import generate_single_reprint")
        print("  generate_single_reprint('reprints.csv', 'reprint_SBS6', 'SBS6_seamless.png')")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    generate_all_reprints(csv_path, output_pdf, output_dir)


if __name__ == "__main__":
    main()