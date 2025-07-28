import numpy as np
import pandas as pd
from reprint_tool.core import normalize, calculate_rmse, calculate_cosine, reprint

def test_normalize():
    x = np.array([0, 1, 2, 3, 4])
    norm = normalize(x)
    assert np.allclose(norm, [0., 0.25, 0.5, 0.75, 1.])

def test_calculate_rmse():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 4])
    assert np.isclose(calculate_rmse(x, y), 0.1, atol=0.2)

def test_calculate_cosine():
    x = np.array([1, 0])
    y = np.array([0, 1])
    assert np.isclose(calculate_cosine(x, y), 1.0)

def test_reprint():
    data = pd.DataFrame({
        'SigA': [0.1, 0.2, 0.3],
        'SigB': [0.4, 0.5, 0.1]
    }, index=[
        'A[C>T]G',
        'A[C>A]G',
        'A[C>G]G'
    ])
    result = reprint(data)
    assert not result.isnull().values.any()
    assert set(result.columns) == {'SigA', 'SigB'}