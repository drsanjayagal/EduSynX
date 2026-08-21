#!/usr/bin/env python3
"""Generate a very large synthetic student dataset in chunked CSV files."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

import config


def generate_regions(fake: Faker, n: int = 5, seed: int = 42) -> list:
    """Return n unique region names using Faker."""
    fake.seed_instance(seed)
    regions = []
    seen = set()
    for _ in range(100):
        state = fake.state()
        if state not in seen:
            seen.add(state)
            regions.append(state)
        if len(regions) == n:
            break

    # Fallback if Faker does not produce enough unique states
    fallback = ["North", "South", "East", "West", "Central"]
    for name in fallback:
        if len(regions) >= n:
            break
        if name not in seen:
            regions.append(name)
    return regions[:n]


def generate_chunk(n: int, rng: np.random.Generator, regions: list, start_id: int) -> pd.DataFrame:
    """Generate a single chunk of synthetic student data with realistic dependencies."""
    # Demographics
    ses_code = rng.choice([0, 1, 2], size=n, p=[0.35, 0.50, 0.15])  # 0=Low,1=Middle,2=High
    ped_code = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.40, 0.20, 0.05])
    gender_code = rng.choice([0, 1, 2], size=n, p=[0.48, 0.48, 0.04])
    region_code = rng.integers(0, len(regions), size=n)

    socioeconomic_status = np.array(["Low", "Middle", "High"])[ses_code]
    parent_education = np.array(["High School", "Bachelors", "Masters", "PhD"])[ped_code]
    gender = np.array(["Male", "Female", "Other"])[gender_code]
    region = np.array(regions)[region_code]

    # Latent psychological/academic traits
    academic_ability = rng.normal(0, 1, size=n) + 0.3 * ses_code + 0.15 * ped_code + rng.normal(0, 0.3, size=n)
    conscientiousness = rng.normal(0, 1, size=n)
    motivation = rng.normal(0, 1, size=n) + 0.2 * ses_code
    mental_health = rng.normal(0, 1, size=n) - 0.1 * ses_code + rng.normal(0, 0.2, size=n)

    semester = rng.integers(1, 9, size=n)
    age = 17 + semester + rng.integers(0, 6, size=n)

    # Academic features
    study_hours = np.clip(
        10 + 5 * conscientiousness + 3 * motivation + 2 * academic_ability + rng.normal(0, 2, size=n),
        0, 60,
    )
    attendance = np.clip(
        65 + 8 * conscientiousness + 3 * motivation + 2 * academic_ability + rng.normal(0, 5, size=n),
        30, 100,
    )
    previous_gpa = np.clip(5 + 1.5 * academic_ability + 0.3 * ses_code + rng.normal(0, 0.6, size=n), 0, 10)
    internal_marks = np.clip(
        35 + 15 * academic_ability + 8 * conscientiousness + 5 * motivation + 0.4 * previous_gpa + rng.normal(0, 8, size=n),
        0, 100,
    )
    assignments_score = np.clip(
        40 + 14 * academic_ability + 10 * conscientiousness + 6 * motivation + 0.3 * study_hours + rng.normal(0, 8, size=n),
        0, 100,
    )
    credit_load = rng.integers(12, 22, size=n)
    backlog_count = rng.poisson(lam=np.clip(1.5 - 0.5 * academic_ability, 0.1, 5), size=n).clip(0, 20)
    online_learning_hours = np.clip(rng.normal(5 + 2 * motivation + 1 * conscientiousness, 3, size=n), 0, 40)
    library_usage_hours = np.clip(rng.normal(3 + 1.5 * conscientiousness, 2, size=n), 0, 30)

    # Behavioral features
    LMS_usage_time = np.clip(rng.normal(8 + 3 * motivation + 2 * conscientiousness, 4, size=n), 0, 60)
    login_frequency = rng.poisson(
        lam=np.clip(15 + 5 * motivation + 3 * conscientiousness, 1, 50), size=n
    ).clip(0, 100)
    submission_delay = np.clip(
        rng.normal(1.5 - 0.5 * conscientiousness - 0.2 * motivation, 1.2, size=n), 0, 15
    )
    forum_participation = rng.poisson(
        lam=np.clip(3 + 2 * motivation + 1 * conscientiousness, 0.5, 20), size=n
    ).clip(0, 50)
    video_watch_time = np.clip(rng.normal(4 + 2 * motivation, 3, size=n), 0, 40)
    assignment_completion_rate = np.clip(
        rng.normal(75 + 10 * conscientiousness + 5 * motivation, 15, size=n), 0, 100
    )
    extra_curricular_hours = np.clip(rng.normal(2 + 1 * conscientiousness, 2, size=n), 0, 30)

    # Psychological features
    stress_level = np.clip(
        np.rint(rng.normal(5 - 0.5 * mental_health + 0.3 * credit_load, 1.8, size=n)), 1, 10
    ).astype(int)
    anxiety_score = np.clip(
        np.rint(rng.normal(5 - 0.8 * mental_health + 0.4 * stress_level, 1.5, size=n)), 1, 10
    ).astype(int)
    motivation_score = np.clip(np.rint(5 + 3 * motivation + rng.normal(0, 1, size=n)), 1, 10).astype(int)
    self_efficacy_score = np.clip(
        np.rint(5 + 2 * academic_ability + 1 * conscientiousness + rng.normal(0, 1, size=n)), 1, 10
    ).astype(int)
    peer_support_score = rng.integers(1, 11, size=n)
    family_support_score = np.clip(rng.integers(1, 11, size=n) + ses_code, 1, 10).astype(int)
    sleep_hours = np.clip(rng.normal(7 - 0.3 * stress_level + 0.2 * mental_health, 1.2, size=n), 3, 12)
    physical_activity_hours = np.clip(rng.normal(2 + 0.5 * mental_health, 1.5, size=n), 0, 20)

    # Institutional features
    course_difficulty = np.clip(rng.normal(5 + 0.3 * semester, 1.5, size=n), 1, 10)
    faculty_rating = np.clip(rng.normal(3.5 + 0.2 * conscientiousness, 0.8, size=n), 1, 5)
    class_size = rng.integers(20, 201, size=n)
    lab_availability_hours = rng.integers(0, 25, size=n)
    internet_access = (rng.random(n) < (0.75 + 0.1 * ses_code)).astype(int)
    scholarship_prob = np.clip(0.15 + 0.05 * academic_ability + 0.1 * ses_code, 0.05, 0.9)
    scholarship = (rng.random(n) < scholarship_prob).astype(int)
    campus_distance = np.clip(rng.normal(10 + 5 * ses_code, 8, size=n), 0, 100)
    tutoring_sessions = rng.poisson(
        lam=np.clip(2 + 0.5 * academic_ability + 0.3 * ses_code, 0, 10), size=n
    ).clip(0, 20)
    part_time_job_hours = np.clip(rng.normal(10 - 4 * ses_code, 5, size=n), 0, 40)

    # Derived performance/engagement/risk scores
    performance_index = np.clip(
        0.4 * internal_marks + 0.3 * assignments_score + 0.2 * previous_gpa * 10 + 0.1 * attendance
        + rng.normal(0, 5, size=n),
        0, 100,
    )
    engagement_score = np.clip(
        0.25 * LMS_usage_time + 0.2 * login_frequency + 0.15 * forum_participation
        + 0.1 * video_watch_time + 0.2 * assignment_completion_rate + 0.1 * online_learning_hours
        + rng.normal(0, 5, size=n),
        0, 100,
    )
    consistency_score = np.clip(
        0.3 * attendance + 0.3 * assignment_completion_rate - 0.2 * submission_delay * 10
        + 0.2 * login_frequency + rng.normal(0, 5, size=n),
        0, 100,
    )
    risk_score = np.clip(
        0.25 * stress_level * 10 + 0.25 * submission_delay * 5 + 0.2 * backlog_count * 10
        + 0.2 * (100 - attendance) + 0.1 * (100 - internal_marks) + rng.normal(0, 5, size=n),
        0, 100,
    )
    improvement_trend = np.clip(
        0.5 * motivation_score * 10 + 0.3 * self_efficacy_score * 10 - 0.2 * stress_level * 10
        + 0.3 * study_hours + rng.normal(0, 10, size=n),
        0, 100,
    )

    # Target: final_result with class imbalance
    success_score = np.clip(
        0.35 * performance_index + 0.25 * engagement_score + 0.2 * consistency_score
        + 0.2 * (100 - risk_score) + 0.1 * previous_gpa * 10 + rng.normal(0, 8, size=n),
        0, 100,
    )
    final_result = np.select(
        [success_score >= 75, success_score >= 55, success_score >= 35],
        ["Distinction", "Pass", "Fail"],
        default="Dropout",
    )

    student_id = start_id + np.arange(n, dtype=np.int64) + 1

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "age": age,
            "gender": gender,
            "region": region,
            "socioeconomic_status": socioeconomic_status,
            "parent_education": parent_education,
            "study_hours": study_hours,
            "attendance": attendance,
            "internal_marks": internal_marks,
            "previous_gpa": previous_gpa,
            "assignments_score": assignments_score,
            "credit_load": credit_load,
            "backlog_count": backlog_count,
            "semester": semester,
            "online_learning_hours": online_learning_hours,
            "library_usage_hours": library_usage_hours,
            "LMS_usage_time": LMS_usage_time,
            "login_frequency": login_frequency,
            "submission_delay": submission_delay,
            "forum_participation": forum_participation,
            "video_watch_time": video_watch_time,
            "assignment_completion_rate": assignment_completion_rate,
            "extra_curricular_hours": extra_curricular_hours,
            "stress_level": stress_level,
            "anxiety_score": anxiety_score,
            "motivation_score": motivation_score,
            "self_efficacy_score": self_efficacy_score,
            "peer_support_score": peer_support_score,
            "family_support_score": family_support_score,
            "sleep_hours": sleep_hours,
            "physical_activity_hours": physical_activity_hours,
            "course_difficulty": course_difficulty,
            "faculty_rating": faculty_rating,
            "class_size": class_size,
            "lab_availability_hours": lab_availability_hours,
            "internet_access": internet_access,
            "scholarship": scholarship,
            "campus_distance": campus_distance,
            "tutoring_sessions": tutoring_sessions,
            "part_time_job_hours": part_time_job_hours,
            "performance_index": performance_index,
            "engagement_score": engagement_score,
            "consistency_score": consistency_score,
            "risk_score": risk_score,
            "improvement_trend": improvement_trend,
            "final_result": final_result,
        }
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic student dataset.")
    parser.add_argument("--rows", type=int, default=config.DEFAULT_TOTAL_ROWS)
    parser.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--output-dir", type=str, default=str(config.DATA_DIR))
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--output-mode", choices=["multiple", "single"], default="multiple")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing data files before generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for pattern in ["student_chunk_*.csv", "student_data.csv"]:
            for file in output_dir.glob(pattern):
                file.unlink()

    fake = Faker()
    regions = generate_regions(fake)

    n_chunks = int(np.ceil(args.rows / args.chunk_size))
    print(f"Generating {args.rows} rows in {n_chunks} chunks of {args.chunk_size}...")

    csv_path = output_dir / "student_data.csv"

    for chunk_idx in tqdm(range(n_chunks), desc="Generating chunks"):
        n = min(args.chunk_size, args.rows - chunk_idx * args.chunk_size)
        rng = np.random.default_rng(args.seed + chunk_idx)
        start_id = chunk_idx * args.chunk_size
        df_chunk = generate_chunk(n, rng, regions, start_id)

        if args.output_mode == "multiple":
            path = output_dir / f"student_chunk_{chunk_idx:05d}.csv"
            df_chunk.to_csv(path, index=False, float_format="%.4f")
        else:
            if chunk_idx == 0 or not csv_path.exists():
                df_chunk.to_csv(csv_path, mode="w", header=True, index=False, float_format="%.4f")
            else:
                df_chunk.to_csv(csv_path, mode="a", header=False, index=False, float_format="%.4f")

    print(f"Data generation complete. Files saved to {output_dir}")


if __name__ == "__main__":
    main()