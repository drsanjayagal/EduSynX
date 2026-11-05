#!/usr/bin/env python3
"""
Generate EduSynX synthetic student dataset.

Revisions addressing reviewers:
  R1-#2: implement Eq.(1) latent covariance and Table 1 equations.
  R1-#3: regenerate descriptive statistics from THIS implementation.
  R1-#4: class distribution is calibrated empirically from the
         success-score distribution (not set by arbitrary thresholds).
  R2-1 : add non-linear feature mappings, structural misspecification and
         an unobserved confounder U so that the target is not recoverable
         by linear inversion of the four observed latent traits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Latent structure (Eq. 1)
# ---------------------------------------------------------------------------
SIGMA_Z = np.array([
    [1.0, 0.3, 0.4, 0.2],
    [0.3, 1.0, 0.5, 0.3],
    [0.4, 0.5, 1.0, 0.4],
    [0.2, 0.3, 0.4, 1.0],
])

# Target class proportions (manuscript: 20 / 40 / 25 / 15)
TARGET_DISTRIBUTION = {  # name: probability
    "Dropout":     0.15,
    "Fail":        0.25,
    "Pass":        0.40,
    "Distinction": 0.20,
}


def generate_regions(fake: Faker, n: int = 5, seed: int = 42) -> list:
    fake.seed_instance(seed)
    regions, seen = [], set()
    for _ in range(100):
        s = fake.state()
        if s not in seen:
            seen.add(s)
            regions.append(s)
        if len(regions) == n:
            break
    for name in ["North", "South", "East", "West", "Central"]:
        if len(regions) >= n:
            break
        if name not in seen:
            regions.append(name)
    return regions[:n]


# ---------------------------------------------------------------------------
# Pilot calibration of target thresholds (R1-#4)
# ---------------------------------------------------------------------------
def _nonlinear_features(A, C, M, H, rng):
    """Compute the four held-out composites + success score.

    Non-linear terms break the linear invertibility of the feature-to-latent
    map (Reviewer 2). The unobserved U adds non-recoverable variance.
    """
    n = len(A)
    eps = lambda s: rng.normal(0.0, s, size=n)

    exam_perf   = 0.50*A + 0.30*C + 0.20*M + 0.25*np.tanh(C*M) + eps(0.8)
    engagement  = 0.40*C + 0.30*M + 0.30*A + 0.20*np.tanh(A*C) + eps(0.8)
    consistency = 0.50*C + 0.30*H + 0.20*M + 0.30*np.tanh(H)   + eps(0.8)
    wellbeing   = 0.60*H + 0.20*M + 0.20*C + 0.30*np.tanh(M-H)+ eps(0.8)

    # Unobserved confounder
    U = rng.normal(0.0, 1.0, size=n)

    success = (0.35*exam_perf + 0.25*engagement
               + 0.20*consistency + 0.20*wellbeing
               + 0.40*U
               + eps(0.6))
    return exam_perf, engagement, consistency, wellbeing, success


def calibrate_thresholds(seed: int = 42, n_pilot: int = 200_000):
    """Estimate tau_1..tau_3 that yield the target class distribution."""
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(np.zeros(4), SIGMA_Z, size=n_pilot)
    A, C, M, H = Z.T
    _, _, _, _, success = _nonlinear_features(A, C, M, H, rng)

    # Cumulative: Dropout (bottom 15%), Fail (next 25% -> 40%),
    #             Pass (next 40% -> 80%), Distinction (top 20%)
    tau1, tau2, tau3 = np.quantile(
        success, [TARGET_DISTRIBUTION["Dropout"],
                  TARGET_DISTRIBUTION["Dropout"] + TARGET_DISTRIBUTION["Fail"],
                  TARGET_DISTRIBUTION["Dropout"] + TARGET_DISTRIBUTION["Fail"]
                    + TARGET_DISTRIBUTION["Pass"]]
    )
    return float(tau1), float(tau2), float(tau3)


# ---------------------------------------------------------------------------
# Single-chunk generator
# ---------------------------------------------------------------------------
def generate_chunk(n: int, rng: np.random.Generator, regions: list,
                   start_id: int, taus: tuple) -> pd.DataFrame:
    tau1, tau2, tau3 = taus

    # --- Demographics ------------------------------------------------------
    ses_code    = rng.choice([0, 1, 2], size=n, p=[0.35, 0.50, 0.15])
    ped_code    = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.40, 0.20, 0.05])
    gender_code = rng.choice([0, 1, 2], size=n, p=[0.48, 0.48, 0.04])
    region_code = rng.integers(0, len(regions), size=n)

    socioeconomic_status = np.array(["Low", "Middle", "High"])[ses_code]
    parent_education     = np.array(["High School", "Bachelors", "Masters", "PhD"])[ped_code]
    gender               = np.array(["Male", "Female", "Other"])[gender_code]
    region               = np.array(regions)[region_code]

    # --- Latents (Eq. 1) + unobserved U ----------------------------------
    Z = rng.multivariate_normal(np.zeros(4), SIGMA_Z, size=n)
    A, C, M, H = Z.T
    U = rng.normal(0.0, 1.0, size=n)  # never used in any observable feature

    semester = rng.integers(1, 9, size=n)
    age      = 17 + semester + rng.integers(0, 6, size=n)

    # --- Academic (Table 1 + non-linear tweaks) -------------------------
    study_hours      = np.clip(10 + 3*C + 2*M + 1.5*A + 0.5*np.tanh(C*M) + rng.normal(0, 2.0, n), 0, 60)
    attendance       = np.clip(70 + 8*np.tanh(C) + 5*M + 3*H + 2*A + rng.normal(0, 5.0, n), 30, 100)
    internal_marks   = np.clip(50 + 10*A + 5*C + 3*M + 2*np.tanh(A*C) + rng.normal(0, 10.0, n), 0, 100)
    previous_gpa     = np.clip(6.0 + 1.0*A + 0.5*C + 0.3*M + rng.normal(0, 0.5, n), 0, 10)
    assignment_score = np.clip(60 + 8*A + 6*C + 4*M + rng.normal(0, 8.0, n), 0, 100)
    credit_load      = rng.integers(12, 22, size=n)
    backlog_count    = rng.poisson(np.clip(1.5 - 0.5*A, 0.05, 5), size=n).clip(0, 20)
    online_learning_hours = np.clip(rng.normal(5 + 2*M + 1*C, 3, n), 0, 40)
    library_usage_hours   = np.clip(rng.normal(3 + 1.5*C, 2, n), 0, 30)
    lab_attendance        = np.clip(rng.normal(70 + 6*C + 4*M, 8, n), 0, 100)
    tutorial_attendance   = np.clip(rng.normal(75 + 5*C + 4*M, 8, n), 0, 100)

    # --- Behavioural (Table 1) -------------------------------------------
    lms_usage_hours           = np.clip(5.0 + 2*C + 1.5*M + 0.5*A + rng.normal(0, np.sqrt(2.0), n), 0, 60)
    login_frequency           = rng.poisson(np.clip(10 + 3*C + 2*M, 1, 50), size=n).clip(0, 100)
    submission_delay          = np.clip(3.0 - 1*C - 0.5*M + rng.normal(0, np.sqrt(1.5), n), 0, 20)
    forum_participation       = rng.poisson(np.clip(5.0 + 1.5*M + 1*C, 0.5, 20), size=n).clip(0, 50)
    video_watch_time          = np.clip(10 + 3*C + 2*M + 0.5*A + rng.normal(0, np.sqrt(6.0), n), 0, 40)
    assignment_completion_rate = np.clip(0.70 + 0.10*C + 0.05*M + rng.normal(0, 0.08, n), 0, 1)
    extra_curricular_hours    = np.clip(rng.normal(3.0 + 0.5*M + 0.3*C, 1.5, n), 0, 30)
    peer_interaction_score    = np.clip(rng.normal(5 + 0.8*M, 1.5, n), 0, 10)

    # --- Psychological (Table 1) -----------------------------------------
    stress_level         = np.clip(np.rint(rng.normal(5.0 - 2*H - 1*M + 1*A, 1.0, n)), 1, 10).astype(int)
    anxiety_score        = np.clip(np.rint(rng.normal(5.0 - 2*H - 1.5*M, 1.2, n)), 1, 10).astype(int)
    motivation_score     = np.clip(np.rint(rng.normal(5.0 + 2*M + 1*C, 1.0, n)), 1, 10).astype(int)
    self_efficacy_score  = np.clip(np.rint(rng.normal(5.0 + 1.5*A + 1.5*C, 1.1, n)), 1, 10).astype(int)
    peer_support_score   = np.clip(np.rint(rng.normal(5.0 + 1*M + 0.5*H, 1.2, n)), 1, 10).astype(int)
    family_support_score = np.clip(np.rint(rng.normal(5.0 + 0.5*H + 0.5*C, 1.4, n)), 1, 10).astype(int)
    sleep_hours          = np.clip(rng.normal(7.0 + 1*H - 0.5*C, np.sqrt(0.5), n), 3, 12)
    physical_activity_hours = np.clip(rng.normal(3.0 + 0.5*H + 0.3*C, 1.0, n), 0, 20)

    # --- Institutional ---------------------------------------------------
    course_difficulty     = np.clip(rng.normal(5 + 0.3*semester, 1.5, n), 1, 10)
    faculty_rating        = np.clip(rng.normal(3.5 + 0.2*C, 0.8, n), 1, 5)
    class_size            = rng.integers(20, 201, size=n)
    lab_availability_hours = rng.integers(0, 25, size=n)
    internet_access       = (rng.random(n) < (0.75 + 0.1*ses_code)).astype(int)
    scholarship           = (rng.random(n) < np.clip(0.15 + 0.05*A + 0.1*ses_code, 0.05, 0.9)).astype(int)
    campus_distance       = np.clip(rng.normal(10 + 5*ses_code, 8, n), 0, 100)
    tutoring_sessions     = rng.poisson(np.clip(2 + 0.5*A + 0.3*ses_code, 0, 10), size=n).clip(0, 20)
    part_time_job_hours   = np.clip(rng.normal(10 - 4*ses_code, 5, n), 0, 40)
    library_resources     = rng.integers(1, 6, size=n)
    academic_advising_hours = np.clip(rng.normal(2 + 0.5*C, 1.5, n), 0, 20)

    # --- Non-circular target --------------------------------------------
    exam_perf, engagement, consistency, wellbeing, success = _nonlinear_features(A, C, M, H, rng)

    final_result = np.select(
        [success > tau3, success > tau2, success > tau1],
        ["Distinction", "Pass", "Fail"],
        default="Dropout",
    )

    student_id = start_id + np.arange(n, dtype=np.int64) + 1

    return pd.DataFrame({
        "student_id": student_id,
        "age": age, "gender": gender, "region": region,
        "socioeconomic_status": socioeconomic_status,
        "parent_education": parent_education,
        "study_hours": study_hours, "attendance": attendance,
        "internal_marks": internal_marks, "previous_gpa": previous_gpa,
        "assignment_score": assignment_score, "credit_load": credit_load,
        "backlog_count": backlog_count, "semester": semester,
        "online_learning_hours": online_learning_hours,
        "library_usage_hours": library_usage_hours,
        "lab_attendance": lab_attendance, "tutorial_attendance": tutorial_attendance,
        "lms_usage_hours": lms_usage_hours, "login_frequency": login_frequency,
        "submission_delay": submission_delay,
        "forum_participation": forum_participation,
        "video_watch_time": video_watch_time,
        "assignment_completion_rate": assignment_completion_rate,
        "extra_curricular_hours": extra_curricular_hours,
        "peer_interaction_score": peer_interaction_score,
        "stress_level": stress_level, "anxiety_score": anxiety_score,
        "motivation_score": motivation_score, "self_efficacy_score": self_efficacy_score,
        "peer_support_score": peer_support_score,
        "family_support_score": family_support_score,
        "sleep_hours": sleep_hours,
        "physical_activity_hours": physical_activity_hours,
        "course_difficulty": course_difficulty, "faculty_rating": faculty_rating,
        "class_size": class_size, "lab_availability_hours": lab_availability_hours,
        "internet_access": internet_access, "scholarship": scholarship,
        "campus_distance": campus_distance, "tutoring_sessions": tutoring_sessions,
        "part_time_job_hours": part_time_job_hours,
        "library_resources": library_resources,
        "academic_advising_hours": academic_advising_hours,
        "final_result": final_result,
    })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=config.DEFAULT_TOTAL_ROWS)
    p.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    p.add_argument("--output-dir", type=str, default=str(config.DATA_DIR))
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    p.add_argument("--output-mode", choices=["multiple", "single"], default="multiple")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for pat in ["student_chunk_*.csv", "student_data.csv"]:
            for f in out.glob(pat):
                f.unlink()

    print("Calibrating target thresholds on 200k pilot sample...")
    taus = calibrate_thresholds(seed=args.seed)
    print(f"  tau1, tau2, tau3 = {taus}")
    (out / "target_thresholds.json").write_text(json.dumps(
        {"tau1": taus[0], "tau2": taus[1], "tau3": taus[2],
         "target_distribution": TARGET_DISTRIBUTION}, indent=2))

    regions = generate_regions(Faker())
    n_chunks = int(np.ceil(args.rows / args.chunk_size))
    csv_path = out / "student_data.csv"

    for i in tqdm(range(n_chunks), desc="Generating"):
        n = min(args.chunk_size, args.rows - i * args.chunk_size)
        rng = np.random.default_rng(args.seed + i)
        df = generate_chunk(n, rng, regions, i * args.chunk_size, taus)
        if args.output_mode == "multiple":
            df.to_csv(out / f"student_chunk_{i:05d}.csv", index=False, float_format="%.4f")
        else:
            df.to_csv(csv_path, mode="w" if i == 0 else "a",
                      header=(i == 0), index=False, float_format="%.4f")

    print(f"Done. Files in {out}")


if __name__ == "__main__":
    main()