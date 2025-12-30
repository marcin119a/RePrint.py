import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from reprint_tool.plot import save_all_signatures_to_pdf, create_main_dashboard
from reprint_tool.core import reprint
from reprint_tool.analyze import (
    create_heatmap_with_custom_sim,
    calculate_rmse,
    calculate_cosine,
)


def main():
    parser = argparse.ArgumentParser(description="Generate PDF plots for all signatures in a DataFrame.")
    parser.add_argument('--input', required=True, help='Path to input CSV/TSV file (with signatures as columns)')
    parser.add_argument('--output_dir', default='output', help='Directory to save PDF files (default: output)')
    parser.add_argument('--prefix', default='reprint_', help='Prefix for output PDF files (default: reprint_)')
    parser.add_argument('--sep', default='\t', help='Column separator in input file (default: tab)')
    parser.add_argument('--save_reprint', default=None, help='If set, save computed reprint DataFrame to this file (CSV/TSV)')
    parser.add_argument('--export_png', action='store_true', help='Also export per-signature plots to PNG files')
    parser.add_argument('--analyze_png', default=None, help='If set, generate similarity heatmap and save to this PNG path')
    parser.add_argument('--analyze_metric', default='rmse', choices=['rmse', 'cosine'], help='Similarity metric for heatmap')
    parser.add_argument('--analyze_colorscale', default='Blues', help='Colorscale for analyze heatmap')
    parser.add_argument('--analyze_hide_heatmap', action='store_true', help='Hide heatmap, show only dendrograms')
    parser.add_argument('--analyze_method', default='complete', help='Linkage method for clustering (e.g., complete, average)')
    args = parser.parse_args()
    
    # Convert escape sequences like '\t' to actual tab character
    sep = args.sep.encode().decode('unicode_escape')

    df = pd.read_csv(args.input, sep=sep, index_col=0)
    df_reprint = reprint(df)

    if args.save_reprint:
        os.makedirs(os.path.dirname(args.save_reprint) or '.', exist_ok=True)
        df_reprint.to_csv(args.save_reprint, sep=sep)
        print(f"Saved reprint DataFrame to {args.save_reprint}")

   
    save_all_signatures_to_pdf(df_reprint, output_dir=args.output_dir, prefix=args.prefix)

    if args.export_png:
        os.makedirs(args.output_dir, exist_ok=True)
        for signature in df_reprint.columns:
            fig = create_main_dashboard(
                df_reprint,
                signature=signature,
                title=f"{args.prefix}{signature} - Probabilities of Specific Tri-nucleotide Context Mutations by Mutation Type",
                yaxis_title="Probabilities",
            )
            png_path = os.path.join(args.output_dir, f"{args.prefix}{signature}.png")
            fig.savefig(png_path, format="png", dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {png_path}")

    if args.analyze_png:
        metric_func = calculate_rmse if args.analyze_metric == 'rmse' else calculate_cosine
        fig = create_heatmap_with_custom_sim(
            df_reprint,
            calc_func=metric_func,
            colorscale=args.analyze_colorscale,
            hide_heatmap=args.analyze_hide_heatmap,
            method=args.analyze_method,
        )
        out_path = args.analyze_png
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved analyze heatmap: {out_path}")


if __name__ == "__main__":
    main()