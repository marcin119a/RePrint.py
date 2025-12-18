import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_main_dashboard(df, signature, title, yaxis_title):
    """
    Original vertical bar chart implementation.
    """
    frequencies = df[signature] * 1

    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    contexts = [f'{x}[{m}]{y}' for m in mutations for x in bases for y in bases]

    colors = {
        'C>A': 'blue',
        'C>G': 'black',
        'C>T': 'red',
        'T>A': 'gray',
        'T>C': 'green',
        'T>G': 'pink'
    }

    fig = go.Figure()
    
    for mutation in mutations:
        mutation_contexts = [c for c in contexts if f'[{mutation}]' in c]
        mutation_frequencies = [frequencies[mc] if mc in frequencies.index else 0 for mc in mutation_contexts]
        
        fig.add_trace(go.Bar(
            x=mutation_contexts,
            y=mutation_frequencies,
            name=mutation,
            marker_color=colors[mutation]
        ))

    y_max = frequencies.max()

    fig.update_layout(
        title=title,
        xaxis_title='Mutation Context',
        yaxis_title=yaxis_title,
        xaxis_tickangle=-90,
        template='plotly_white',
        barmode='group',
        legend_title='Mutation Type',
        yaxis_range=[0, y_max], 
        margin=dict(l=50, r=50, t=50, b=150),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=10))
    )

    return fig


def create_main_dashboard_horizontal(df, signature, title, yaxis_title):
    """
    Horizontal bar chart with 3 rows, each showing a pair of complementary mutations.
    Each row has 2 groups of 16 bars (C-centered and T-centered contexts).
    Rows in order:
    - C>T (red, left 16) + T>G (light pink, right 16)
    - C>G (black, left 16) + T>C (light green, right 16)
    - C>A (light blue, left 16) + T>A (gray, right 16)
    
    X-axis shows 32 trinucleotide contexts: first 16 (ACA-TCT) then last 16 (ATA-TTT)
    """
    frequencies = df[signature] * 1

    # Define the order of 32 trinucleotide contexts for X-axis
    # First 16: C-centered contexts
    c_contexts = [
        'ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT',
        'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT'
    ]
    # Last 16: T-centered contexts
    t_contexts = [
        'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT',
        'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT'
    ]
    # Combined order for X-axis
    context_order = c_contexts + t_contexts

    # Define mutation pairs: (C-mutation, T-mutation) for each row
    mutation_pairs = [
        ('C>T', 'T>G'),  # Top row
        ('C>G', 'T>C'),  # Middle row
        ('C>A', 'T>A')   # Bottom row
    ]
    
    def get_value_for_context(context_short, mutation_type):
        """
        Get value for a short context (e.g., 'ACA') and mutation type (e.g., 'C>A').
        Converts 'ACA' + 'C>A' -> 'A[C>A]A'
        """
        full_context = f'{context_short[0]}[{mutation_type}]{context_short[2]}'
        return frequencies[full_context] if full_context in frequencies.index else 0

    # Color scheme for each mutation pair
    color_schemes = {
        ('C>T', 'T>G'): {
            'left': 'red',        # C>T contexts (left 16)
            'right': 'lightpink'  # T>G contexts (right 16)
        },
        ('C>G', 'T>C'): {
            'left': 'black',      # C>G contexts (left 16)
            'right': 'lightgreen' # T>C contexts (right 16)
        },
        ('C>A', 'T>A'): {
            'left': 'lightblue',  # C>A contexts (left 16)
            'right': 'gray'       # T>A contexts (right 16)
        }
    }

    # Create subplots with 3 rows, one for each mutation pair
    fig = make_subplots(
        rows=len(mutation_pairs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['', '', '']  # No titles for subplots
    )

    # Get overall min/max for consistent y-axis scaling
    all_values = []
    for c_mut, t_mut in mutation_pairs:
        for ctx in c_contexts:
            val = get_value_for_context(ctx, c_mut)
            all_values.append(val)
        for ctx in t_contexts:
            val = get_value_for_context(ctx, t_mut)
            all_values.append(val)
    
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = max(abs(y_min), abs(y_max)) * 1.1  # Add 10% padding

    for row_idx, (c_mut, t_mut) in enumerate(mutation_pairs, 1):
        # Get values for C-mutation contexts (first 16)
        c_values = [get_value_for_context(ctx, c_mut) for ctx in c_contexts]
        # Get values for T-mutation contexts (last 16)
        t_values = [get_value_for_context(ctx, t_mut) for ctx in t_contexts]
        
        # Rotate upper two rows (multiply by -1)
        if row_idx <= 2:
            c_values = [-v for v in c_values]
            t_values = [-v for v in t_values]
        
        # Make bottom row symmetric (values can be negative)
        # Bottom row already has both positive and negative values, so no change needed
        
        colors = color_schemes[(c_mut, t_mut)]
        
        # Add trace for C-mutation contexts (left 16 bars)
        fig.add_trace(
            go.Bar(
                x=c_contexts,
                y=c_values,
                name=c_mut,
                marker_color=colors['left'],
                showlegend=True,
                orientation='v',
                width=0.9  # Narrower bars
            ),
            row=row_idx,
            col=1
        )
        
        # Add trace for T-mutation contexts (right 16 bars)
        fig.add_trace(
            go.Bar(
                x=t_contexts,
                y=t_values,
                name=t_mut,
                marker_color=colors['right'],
                showlegend=True,
                orientation='v',
                width=0.9  # Narrower bars
            ),
            row=row_idx,
            col=1
        )
        
        # Add horizontal line at y=0 for baseline
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_width=1,
            line_color="black",
            opacity=0.3,
            row=row_idx,
            col=1
        )
        
        # Update y-axis for this subplot - hide axes, symmetric range
        fig.update_yaxes(
            range=[-y_range, y_range],
            title_text="",
            showticklabels=False,  # Hide tick labels
            showgrid=False,  # Hide grid lines
            showline=False,  # Hide axis line
            zeroline=False,  # Hide zero line
            row=row_idx,
            col=1
        )

    # Update x-axis (only for bottom subplot) - hide axes
    fig.update_xaxes(
        title_text="",
        tickangle=-90,
        tickfont=dict(size=8),
        tickmode='array',
        tickvals=context_order,
        ticktext=context_order,
        showgrid=False,  # Hide grid lines
        showline=False,  # Hide axis line
        row=len(mutation_pairs),
        col=1
    )
    
    # Hide x-axis for upper rows
    for row_idx in range(1, len(mutation_pairs)):
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            showline=False,
            row=row_idx,
            col=1
        )

    # Update layout
    fig.update_layout(
        template='plotly_white',
        height=600,
        margin=dict(l=80, r=50, t=50, b=200),
        barmode='group',
        bargap=0.01,  # Minimal gap between groups of bars
        bargroupgap=0.01,  # Minimal gap between bars within a group
        legend=dict(
            title="Mutation Type",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig

import plotly.io as pio
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import matplotlib.pyplot as plt
import io

def save_all_signatures_to_pdf(df, output_dir=".", prefix="reprint_", yaxis_title="Probabilities"):
    """
    Creates and saves PDF plots for all signatures in the DataFrame.
    Each signature is saved as a separate PDF file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    for signature in df.columns:
        fig = create_main_dashboard(
            df,
            signature=signature,
            title=f"{prefix}{signature} - Probabilities of Specific Tri-nucleotide Context Mutations",
            yaxis_title=yaxis_title
        )
        pdf_path = os.path.join(output_dir, f"{prefix}{signature}.pdf")
        fig.write_image(pdf_path, format="pdf")
        print(f"Saved: {pdf_path}")


def save_all_signatures_to_single_pdf(df, output_pdf, prefix="", yaxis_title="Probabilities"):
    """
    Creates and saves all signature plots in a single PDF file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with signatures as columns
    output_pdf : str
        Path to output PDF file
    prefix : str
        Prefix for signature names in titles
    yaxis_title : str
        Title for y-axis
    """
    import os
    import tempfile
    
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate all plots as PNG images first
        image_paths = []
        for i, signature in enumerate(df.columns, 1):
            print(f"  Processing {i}/{len(df.columns)}: {signature}")
            fig = create_main_dashboard(
                df,
                signature=signature,
                title=f"{prefix}{signature} - Probabilities of Specific Tri-nucleotide Context Mutations by Mutation Type",
                yaxis_title=yaxis_title
            )
            
            # Save as high-resolution PNG temporarily
            temp_png = os.path.join(temp_dir, f"{signature}.png")
            try:
                # Use high resolution: scale=3 gives ~3600x2400 pixels (high quality)
                fig.write_image(temp_png, format="png", scale=3)
                image_paths.append(temp_png)
            except Exception as e:
                print(f"    Warning: Could not export {signature} as image: {e}")
                continue
        
        # Combine all images into single PDF using matplotlib with high DPI
        print(f"\nCombining {len(image_paths)} plots into PDF...")
        with PdfPages(output_pdf) as pdf:
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Get actual image DPI and dimensions
                    # Plotly default is 72 DPI, with scale=3 we get 216 DPI effectively
                    # Calculate figure size to maintain aspect ratio at 300 DPI
                    img_dpi = 216  # Effective DPI from plotly scale=3
                    img_width_inch = img.width / img_dpi
                    img_height_inch = img.height / img_dpi
                    
                    # Create matplotlib figure matching image size
                    fig_pdf = plt.figure(figsize=(img_width_inch, img_height_inch), dpi=300)
                    ax = fig_pdf.add_subplot(111)
                    ax.imshow(img, aspect='auto', interpolation='bilinear')
                    ax.axis('off')
                    # Save with high DPI for best quality
                    pdf.savefig(fig_pdf, bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close(fig_pdf)
                except Exception as e:
                    print(f"    Warning: Could not add {img_path} to PDF: {e}")
                    continue
        
        print(f"✓ Combined PDF saved: {output_pdf}")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
