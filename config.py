"""Central configuration and column definitions for EduSynX."""
from pathlib import Path

RANDOM_STATE = 42
TARGET = "final_result"
ID_COL = "student_id"

FEATURE_COLS = [
    "age",
    "gender",
    "region",
    "socioeconomic_status",
    "parent_education",
    "study_hours",
    "attendance",
    "internal_marks",
    "previous_gpa",
    "assignments_score",
    "credit_load",
    "backlog_count",
    "semester",
    "online_learning_hours",
    "library_usage_hours",
    "LMS_usage_time",
    "login_frequency",
    "submission_delay",
    "forum_participation",
    "video_watch_time",
    "assignment_completion_rate",
    "extra_curricular_hours",
    "stress_level",
    "anxiety_score",
    "motivation_score",
    "self_efficacy_score",
    "peer_support_score",
    "family_support_score",
    "sleep_hours",
    "physical_activity_hours",
    "course_difficulty",
    "faculty_rating",
    "class_size",
    "lab_availability_hours",
    "internet_access",
    "scholarship",
    "campus_distance",
    "tutoring_sessions",
    "part_time_job_hours",
    "performance_index",
    "engagement_score",
    "consistency_score",
    "risk_score",
    "improvement_trend",
]

CATEGORICAL_COLS = [
    "gender",
    "region",
    "socioeconomic_status",
    "parent_education",
]

NUMERIC_COLS = [col for col in FEATURE_COLS if col not in CATEGORICAL_COLS]

ALL_COLUMNS = [ID_COL] + FEATURE_COLS + [TARGET]

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

DEFAULT_TOTAL_ROWS = 10_000_000
DEFAULT_CHUNK_SIZE = 1_000_000