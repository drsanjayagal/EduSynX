#!/usr/bin/env python3
"""
Decision support: emits model-level scenarios, never causal prescriptions.

Fixes:
  R3-#2, R2-causal-language: replaces imperative 'Increase study hours...'
  with 'Model scenario: with study_hours at X, model predicts Pass'
  plus explicit causal caveat.
"""
# ... (existing imports unchanged) ...

CAVEAT = ("These are model-level scenarios. They describe what the model would "
          "predict if a feature changed. They are not causal recommendations; "
          "causal validation is required before any intervention.")


def emit_insight(student_id, predicted_class, top_shap, row):
    lines = [f"Student {student_id}: predicted class = {predicted_class}",
             "  Model-scenario analysis (NOT a causal prescription):"]
    for feat, shap_val in top_shap:
        current = row.get(feat, "n/a")
        direction = "higher" if shap_val > 0 else "lower"
        lines.append(
            f"   • If {feat} were {direction} than its current value ({current}), "
            f"the model output changes by {shap_val:+.3f} in SHAP units."
        )
    lines.append(f"  ⚠ {CAVEAT}")
    return "\n".join(lines)