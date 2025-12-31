import os
import pandas as pd
import pytest
from pathlib import Path
from reprint_tool.core import reprint
from reprint_tool.plot_static import (
    generate_single_reprint,
    generate_all_reprints,
    load_reprint_data,
    extract_reprint_vector,
    plot_reprint_seamless
)


def test_load_reprint_data(tmp_path):
    """Test loading reprint data from CSV file."""
    # Create test CSV file
    test_data = pd.DataFrame({
        'reprint_SBS1': [0.1, 0.2, 0.3] * 32,  # 96 values for 96 mutation contexts
        'reprint_SBS2': [0.2, 0.3, 0.4] * 32
    }, index=[f'{l}[{m}]{r}' for l in 'ACGT' for m in ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G'] for r in 'ACGT'])
    
    csv_path = tmp_path / "test_reprints.csv"
    test_data.to_csv(csv_path)
    
    # Load data
    df = load_reprint_data(str(csv_path))
    assert df.shape == test_data.shape
    assert list(df.columns) == list(test_data.columns)


def test_extract_reprint_vector(tmp_path):
    """Test extracting a single reprint vector."""
    # Create test CSV file with proper mutation contexts
    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    contexts = [f'{x}[{m}]{y}' for m in mutations for x in bases for y in bases]
    
    test_data = pd.DataFrame({
        'reprint_SBS1': [0.01] * len(contexts)
    }, index=contexts)
    
    csv_path = tmp_path / "test_reprints.csv"
    test_data.to_csv(csv_path)
    
    df = load_reprint_data(str(csv_path))
    c_probs, t_probs = extract_reprint_vector(df, 'reprint_SBS1')
    
    assert c_probs.shape == (16, 3), "C probabilities should be 16x3"
    assert t_probs.shape == (16, 3), "T probabilities should be 16x3"


def test_generate_single_reprint_png(tmp_path):
    """Test that generate_single_reprint creates a valid PNG file."""
    # Create test CSV file with proper mutation contexts
    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    contexts = [f'{x}[{m}]{y}' for m in mutations for x in bases for y in bases]
    
    # Create realistic probabilities that sum to 1 for each context
    test_data = pd.DataFrame({
        'reprint_SBS1': [0.33, 0.33, 0.34] * 32
    }, index=contexts)
    
    csv_path = tmp_path / "test_reprints.csv"
    test_data.to_csv(csv_path)
    
    output_path = tmp_path / "test_reprint_SBS1.png"
    
    # Generate PNG
    generate_single_reprint(
        csv_path=str(csv_path),
        reprint_name='reprint_SBS1',
        output_path=str(output_path)
    )
    
    # Check that file was created
    assert output_path.exists(), "PNG file was not created"
    assert output_path.stat().st_size > 0, "PNG file is empty"


def test_generate_all_reprints_png(tmp_path):
    """Test that generate_all_reprints creates valid PNG files."""
    # Create test CSV file with proper mutation contexts
    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    contexts = [f'{x}[{m}]{y}' for m in mutations for x in bases for y in bases]
    
    # Create realistic probabilities that sum to 1 for each context
    test_data = pd.DataFrame({
        'reprint_SBS1': [0.33, 0.33, 0.34] * 32,
        'reprint_SBS2': [0.25, 0.35, 0.40] * 32,
        'reprint_SBS3': [0.40, 0.30, 0.30] * 32
    }, index=contexts)
    
    csv_path = tmp_path / "test_reprints.csv"
    test_data.to_csv(csv_path)
    
    output_dir = tmp_path / "png_outputs"
    output_dir.mkdir()
    
    # Generate PNGs
    generate_all_reprints(
        csv_path=str(csv_path),
        output_pdf=None,
        output_dir=str(output_dir)
    )
    
    # Check that PNG files were created
    png_files = list(output_dir.glob("reprint_*.png"))
    assert len(png_files) == 3, f"Expected 3 PNG files, got {len(png_files)}"
    
    for png_file in png_files:
        assert png_file.stat().st_size > 0, f"PNG file {png_file} is empty"


def test_generate_all_reprints_pdf_and_png(tmp_path):
    """Test that generate_all_reprints creates both PDF and PNG files."""
    # Create test CSV file with proper mutation contexts
    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    contexts = [f'{x}[{m}]{y}' for m in mutations for x in bases for y in bases]
    
    # Create realistic probabilities that sum to 1 for each context
    test_data = pd.DataFrame({
        'reprint_SBS1': [0.33, 0.33, 0.34] * 32,
        'reprint_SBS2': [0.25, 0.35, 0.40] * 32
    }, index=contexts)
    
    csv_path = tmp_path / "test_reprints.csv"
    test_data.to_csv(csv_path)
    
    output_pdf = tmp_path / "test_all_reprints.pdf"
    output_dir = tmp_path / "png_outputs"
    output_dir.mkdir()
    
    # Generate PDF and PNGs
    generate_all_reprints(
        csv_path=str(csv_path),
        output_pdf=str(output_pdf),
        output_dir=str(output_dir)
    )
    
    # Check that PDF was created
    assert output_pdf.exists(), "PDF file was not created"
    assert output_pdf.stat().st_size > 0, "PDF file is empty"
    
    # Check that PNG files were created
    png_files = list(output_dir.glob("reprint_*.png"))
    assert len(png_files) == 2, f"Expected 2 PNG files, got {len(png_files)}"
    
    for png_file in png_files:
        assert png_file.stat().st_size > 0, f"PNG file {png_file} is empty"


def test_plot_reprint_seamless_creates_figure(tmp_path):
    """Test that plot_reprint_seamless returns a valid matplotlib figure."""
    import numpy as np
    
    # Create test data
    c_probs = np.random.rand(16, 3)
    t_probs = np.random.rand(16, 3)
    
    # Normalize so each column sums to 1.0
    for i in range(16):
        c_probs[i] = c_probs[i] / c_probs[i].sum()
        t_probs[i] = t_probs[i] / t_probs[i].sum()
    
    # Generate figure
    fig = plot_reprint_seamless('test_reprint', c_probs, t_probs)
    
    assert fig is not None, "Figure was not created"
    assert len(fig.axes) == 3, "Figure should have 3 axes (top, mid, bot)"
    
    # Save to verify it works
    output_path = tmp_path / "test_figure.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    assert output_path.exists(), "Figure was not saved"


def test_generate_single_reprint_with_custom_title(tmp_path):
    """Test that generate_single_reprint accepts custom title."""
    # Create test CSV file with proper mutation contexts
    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    contexts = [f'{x}[{m}]{y}' for m in mutations for x in bases for y in bases]
    
    test_data = pd.DataFrame({
        'reprint_SBS1': [0.33, 0.33, 0.34] * 32
    }, index=contexts)
    
    csv_path = tmp_path / "test_reprints.csv"
    test_data.to_csv(csv_path)
    
    output_path = tmp_path / "test_custom_title.png"
    
    # Generate PNG with custom title
    generate_single_reprint(
        csv_path=str(csv_path),
        reprint_name='reprint_SBS1',
        output_path=str(output_path),
        title="Custom Test Title"
    )
    
    assert output_path.exists(), "PNG file was not created"
    assert output_path.stat().st_size > 0, "PNG file is empty"

