import pandas as pd
from reprint_tool.core import reprint
import os
from reprint_tool.plot import save_all_signatures_to_pdf


def test_pdf_and_csv_export(tmp_path):
    # Create test data
    input_file = os.path.join(os.path.dirname(__file__), 'data', 'COSMIC_v1_SBS_GRCh37.txt')
    df = pd.read_csv(input_file, sep='\t', index_col=0)
    df_reprint = reprint(df)

    # Test saving of PDFs (separate files)
    pdf_dir = tmp_path / "pdfs"
    save_all_signatures_to_pdf(df_reprint, output_dir=str(pdf_dir), prefix="test_")
    pdf_files = list(pdf_dir.glob("test_*.pdf"))
    assert len(pdf_files) == len(df_reprint.columns), "Should create one PDF per signature"
    for pdf in pdf_files:
        assert pdf.stat().st_size > 0, f"PDF file {pdf} is empty"

    #  Test saving of CSV/TSV file
    csv_file = tmp_path / "reprint_matrix.tsv"
    df_reprint.to_csv(csv_file, sep='\t')
    assert csv_file.exists(), "CSV/TSV file was not created"
    assert csv_file.stat().st_size > 0, "CSV/TSV file is empty"
    df_loaded = pd.read_csv(csv_file, sep='\t', index_col=0)
    assert df_loaded.shape == df_reprint.shape, "Saved CSV/TSV shape mismatch"

