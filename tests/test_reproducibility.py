"""Same seed -> identical chunk (first 100 rows)."""
import numpy as np
import generate_data, config

def test_same_seed_same_output():
    a = generate_data.generate_chunk(100, np.random.default_rng(42), ["N","S"], 0, (-0.4, 0.2, 0.9))
    b = generate_data.generate_chunk(100, np.random.default_rng(42), ["N","S"], 0, (-0.4, 0.2, 0.9))
    pd.testing.assert_frame_equal(a, b)