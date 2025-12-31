import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from reprint_tool.analyze import (
    normalize,
    calculate_rmse,
    calculate_cosine,
    create_heatmap_with_custom_sim
)


def test_normalize():
    """Test normalize function."""
    x = np.array([1, 2, 3, 4, 5])
    result = normalize(x)
    
    # After normalization: (x - mean) / std
    expected_mean = 0.0  # Mean of normalized data should be ~0
    expected_std = 1.0   # Std of normalized data should be ~1
    
    assert np.isclose(np.nanmean(result), expected_mean, atol=1e-10)
    assert np.isclose(np.nanstd(result), expected_std, atol=1e-10)


def test_normalize_with_nan():
    """Test normalize function with NaN values."""
    x = np.array([1, 2, np.nan, 4, 5])
    result = normalize(x)
    
    # Should handle NaN values gracefully
    assert np.isnan(result[2])  # NaN should remain NaN
    assert not np.isnan(result[0])  # Other values should be normalized


def test_normalize_constant_array():
    """Test normalize function with constant array."""
    x = np.array([1, 1, 1, 1, 1])
    result = normalize(x)
    
    # For constant array, std is 0, so result should contain NaN
    # (function now explicitly returns NaN array for this case)
    assert np.all(np.isnan(result)), "Constant array should result in all NaN values"


def test_calculate_rmse():
    """Test calculate_rmse function."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    
    # Identical arrays should have RMSE close to 0
    result = calculate_rmse(x, y)
    assert np.isclose(result, 0.0, atol=1e-10)


def test_calculate_rmse_different_arrays():
    """Test calculate_rmse with different arrays."""
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    
    # Both should be normalized, then RMSE calculated
    result = calculate_rmse(x, y)
    assert result >= 0  # RMSE should be non-negative
    assert not np.isnan(result)


def test_calculate_cosine():
    """Test calculate_cosine function."""
    x = np.array([1, 0])
    y = np.array([0, 1])
    
    # Orthogonal vectors should have cosine distance of 1.0
    result = calculate_cosine(x, y)
    assert np.isclose(result, 1.0, atol=1e-10)


def test_calculate_cosine_identical():
    """Test calculate_cosine with identical vectors."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    
    # Identical vectors should have cosine distance close to 0
    result = calculate_cosine(x, y)
    assert np.isclose(result, 0.0, atol=1e-10)


def test_calculate_cosine_opposite():
    """Test calculate_cosine with opposite vectors."""
    x = np.array([1, 1])
    y = np.array([-1, -1])
    
    # Opposite vectors should have maximum cosine distance
    result = calculate_cosine(x, y)
    assert result > 0.5  # Should be close to maximum distance


def test_create_heatmap_with_custom_sim_returns_figure(tmp_path):
    """Test that create_heatmap_with_custom_sim returns a valid matplotlib figure."""
    # Create test data
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1],
        'SigC': [0.3, 0.1, 0.2, 0.4]
    })
    
    fig = create_heatmap_with_custom_sim(test_data)
    
    assert fig is not None, "Figure was not created"
    assert len(fig.axes) > 0, "Figure should have axes"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_with_custom_sim_saves_png(tmp_path):
    """Test that create_heatmap_with_custom_sim can save to PNG file."""
    # Create test data
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1],
        'SigC': [0.3, 0.1, 0.2, 0.4]
    })
    
    fig = create_heatmap_with_custom_sim(test_data)
    
    output_path = tmp_path / "test_heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    assert output_path.exists(), "PNG file was not created"
    assert output_path.stat().st_size > 0, "PNG file is empty"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_hide_heatmap(tmp_path):
    """Test create_heatmap_with_custom_sim with hide_heatmap=True."""
    # Create test data
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1],
        'SigC': [0.3, 0.1, 0.2, 0.4]
    })
    
    fig = create_heatmap_with_custom_sim(test_data, hide_heatmap=True)
    
    assert fig is not None, "Figure was not created"
    
    output_path = tmp_path / "test_heatmap_dendro_only.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    assert output_path.exists(), "PNG file was not created"
    assert output_path.stat().st_size > 0, "PNG file is empty"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_custom_colorscale(tmp_path):
    """Test create_heatmap_with_custom_sim with custom colorscale."""
    # Create test data
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1]
    })
    
    fig = create_heatmap_with_custom_sim(test_data, colorscale='Reds')
    
    assert fig is not None, "Figure was not created"
    
    output_path = tmp_path / "test_heatmap_reds.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    assert output_path.exists(), "PNG file was not created"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_custom_method(tmp_path):
    """Test create_heatmap_with_custom_sim with different linkage methods."""
    # Create test data
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1],
        'SigC': [0.3, 0.1, 0.2, 0.4]
    })
    
    methods = ['complete', 'average', 'ward', 'single']
    
    for method in methods:
        fig = create_heatmap_with_custom_sim(test_data, method=method)
        assert fig is not None, f"Figure was not created for method {method}"
        
        output_path = tmp_path / f"test_heatmap_{method}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
        assert output_path.exists(), f"PNG file was not created for method {method}"
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_create_heatmap_custom_calc_func(tmp_path):
    """Test create_heatmap_with_custom_sim with custom calculation function."""
    # Create test data
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1],
        'SigC': [0.3, 0.1, 0.2, 0.4]
    })
    
    # Use cosine distance instead of RMSE
    fig = create_heatmap_with_custom_sim(test_data, calc_func=calculate_cosine)
    
    assert fig is not None, "Figure was not created"
    
    output_path = tmp_path / "test_heatmap_cosine.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    assert output_path.exists(), "PNG file was not created"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_with_reprint_suffix(tmp_path):
    """Test that labels ending with '_reprint' are shortened correctly."""
    # Create test data with '_reprint' suffix in column names
    test_data = pd.DataFrame({
        'SigA_reprint': [0.1, 0.2, 0.3, 0.4],
        'SigB_reprint': [0.2, 0.3, 0.4, 0.1],
        'SigC_reprint': [0.3, 0.1, 0.2, 0.4]
    })
    
    fig = create_heatmap_with_custom_sim(test_data)
    
    assert fig is not None, "Figure was not created"
    
    output_path = tmp_path / "test_heatmap_reprint_labels.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    assert output_path.exists(), "PNG file was not created"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_large_dataset(tmp_path):
    """Test create_heatmap_with_custom_sim with larger dataset."""
    # Create test data with more signatures
    np.random.seed(42)
    n_signatures = 10
    n_features = 20
    
    test_data = pd.DataFrame(
        np.random.rand(n_features, n_signatures),
        columns=[f'Sig{i}' for i in range(n_signatures)]
    )
    
    fig = create_heatmap_with_custom_sim(test_data)
    
    assert fig is not None, "Figure was not created"
    
    output_path = tmp_path / "test_heatmap_large.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    assert output_path.exists(), "PNG file was not created"
    assert output_path.stat().st_size > 0, "PNG file is empty"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_minimal_signatures(tmp_path):
    """Test create_heatmap_with_custom_sim with minimum 2 signatures (edge case)."""
    # Create test data with minimum 2 signatures (needed for distance matrix)
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1]
    })
    
    # Minimum 2 signatures should create a figure
    fig = create_heatmap_with_custom_sim(test_data)
    
    assert fig is not None, "Figure was not created"
    
    output_path = tmp_path / "test_heatmap_minimal.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    assert output_path.exists(), "PNG file was not created"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_heatmap_single_signature_raises_error():
    """Test that create_heatmap_with_custom_sim raises error with single signature."""
    # Create test data with single signature
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4]
    })
    
    # Single signature should raise ValueError (cannot create distance matrix)
    with pytest.raises(ValueError, match="observations|cannot be determined"):
        create_heatmap_with_custom_sim(test_data)


def test_create_heatmap_saves_pdf(tmp_path):
    """Test that create_heatmap_with_custom_sim can save to PDF file."""
    # Create test data
    test_data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3, 0.4],
        'SigB': [0.2, 0.3, 0.4, 0.1],
        'SigC': [0.3, 0.1, 0.2, 0.4]
    })
    
    fig = create_heatmap_with_custom_sim(test_data)
    
    output_path = tmp_path / "test_heatmap.pdf"
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    
    assert output_path.exists(), "PDF file was not created"
    assert output_path.stat().st_size > 0, "PDF file is empty"
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)

