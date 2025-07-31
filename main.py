import argparse
import pandas as pd
from reprint_tool.plot import save_all_signatures_to_pdf
from reprint_tool.core import reprint

def main():
    parser = argparse.ArgumentParser(description="Generate PDF plots for all signatures in a DataFrame.")
    parser.add_argument('--input', required=True, help='Path to input CSV/TSV file (with signatures as columns)')
    parser.add_argument('--output_dir', default='.', help='Directory to save PDF files (default: current directory)')
    parser.add_argument('--prefix', default='reprint_', help='Prefix for output PDF files (default: reprint_)')
    parser.add_argument('--sep', default='\t', help='Column separator in input file (default: tab)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep, index_col=0)
    df_reprint = reprint(df)

    save_all_signatures_to_pdf(df_reprint, output_dir=args.output_dir, prefix=args.prefix)

if __name__ == "__main__":
    main()