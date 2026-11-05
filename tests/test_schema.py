import numpy as np, pytest, generate_data

@pytest.fixture(scope="session")
def sample_df():
    rng = np.random.default_rng(0)
    taus = generate_data.calibrate_thresholds(seed=0, n_pilot=50_000)
    return generate_data.generate_chunk(10_000, rng, ["N","S","E","W","C"], 0, taus)