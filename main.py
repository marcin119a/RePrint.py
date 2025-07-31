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
    parser.add_argument('--save_reprint', default=None, help='If set, save computed reprint DataFrame to this file (CSV/TSV)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep, index_col=0)
    df_reprint = reprint(df)

    if args.save_reprint:
        df_reprint.to_csv(args.save_reprint, sep=args.sep)
        print(f"Saved reprint DataFrame to {args.save_reprint}")

   
    save_all_signatures_to_pdf(df_reprint, output_dir=args.output_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()