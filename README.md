# reprint-tool

**Tool to compute DNA Repair FootPrint (RePrint) from COSMIC signatures and visualize the results as publication-ready PDF plots.**

## Features

- Computes RePrint (DNA repair footprint) from input signature matrices (e.g., COSMIC).
- Generates barplot visualizations for each signature.
- Exports all plots as separate PDF files or as a single multi-page PDF.
- Can save the computed RePrint matrix to a CSV/TSV file.
- Command-line interface for easy batch processing.

## Installation

You need Python 3.8 or newer.

Install dependencies:

```bash
pip install numpy pandas plotly kaleido Pillow
```

For development and testing:

```bash
pip install pytest pytest-cov
```

## Usage

### 1. Compute and plot RePrints, saving each signature as a separate PDF

```bash
python main.py --input path/to/input_signatures.tsv --output_dir pdfs --prefix reprint_ --sep '\t'
```

- `--input` – path to your input file (TSV/CSV, signatures in columns, mutation types in rows)
- `--output_dir` – directory to save PDF files (default: current directory)
- `--prefix` – prefix for output PDF files (default: `reprint_`)
- `--sep` – column separator in input file (default: tab)
- `--save_reprint` – (optional) path to save the computed RePrint matrix as CSV/TSV

### 2. Save all plots in a single multi-page PDF

```bash
python main.py --input path/to/input_signatures.tsv all_signatures.pdf --prefix reprint_ --sep '\t'
```

- `--save_reprint` – (optional) path to save the computed RePrint matrix as CSV/TSV

### 3. Save only the computed RePrint matrix (no plots)

```bash
python main.py --input path/to/input_signatures.tsv --save_reprint reprint_matrix.tsv --sep '\t'
```

## Input Format

Input should be a tab-separated (or CSV) file with mutation types as rows and signatures as columns, e.g.:

```
Type    Signature_1    Signature_2    ...
A[C>A]A 0.011          0.00068        ...
A[C>A]C 0.0091         0.00061        ...
...
```

## Output

- Individual PDF files for each signature (if using `--output_dir`)
- Or a single PDF file with all plots (if using `--single_pdf`)
- The computed RePrint matrix as a CSV/TSV file (if using `--save_reprint`)

## Example

```bash
python main.py --input tests/data/input_COSMIC_v2_SBS_GRCh37.txt --prefix reprint_ --sep '\t' --save_reprint pdfs/reprint_matrix.tsv
```

## Project Structure

```
reprint_tool/
    core.py         # Core computation (RePrint)
    plot.py         # Plotting and PDF export
main.py             # Command-line interface
tests/              # Unit tests and test data
```

