"""R2-leakage: verify composites are NOT in the predictor DataFrame."""
import pandas as pd, numpy as np
import config

def test_no_composites_in_predictors(sample_df):
    for col in config.HELD_OUT_COLS:
        assert col not in sample_df.columns, f"{col} leaked into predictors"
    for col in ["performance_index", "engagement_score", "consistency_score",
                "risk_score", "improvement_trend"]:
        assert col not in sample_df.columns, f"old leakage column {col} present"

def test_class_distribution_close_to_target(sample_df):
    props = sample_df["final_result"].value_counts(normalize=True)
    for cls, target_p in config_target().items():
        assert abs(props.get(cls, 0) - target_p) < 0.03, \
            f"{cls}: got {props.get(cls,0):.3f}, expected ~{target_p}"