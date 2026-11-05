"""Central configuration and column definitions for EduSynX."""
from pathlib import Path

RANDOM_STATE = 42
TARGET = "final_result"
ID_COL = "student_id"

# 44 predictors (Table in manuscript Sec 3.3).
# Removed: performance_index, engagement_score, consistency_score,
#          risk_score, improvement_trend  -> these are held-out composites
#          used to construct the target and must NOT be predictors.
FEATURE_COLS = [
    # Demographic (5)
    "age", "gender", "region", "socioeconomic_status", "parent_education",
    # Academic (12)
    "study_hours", "attendance", "internal_marks", "previous_gpa",
    "assignment_score", "credit_load", "backlog_count", "semester",
    "online_learning_hours", "library_usage_hours",
    "lab_attendance", "tutorial_attendance",
    # Behavioural (8)
    "lms_usage_hours", "login_frequency", "submission_delay",
    "forum_participation", "video_watch_time", "assignment_completion_rate",
    "extra_curricular_hours", "peer_interaction_score",
    # Psychological (8)
    "stress_level", "anxiety_score", "motivation_score", "self_efficacy_score",
    "peer_support_score", "family_support_score", "sleep_hours",
    "physical_activity_hours",
    # Institutional (11)
    "course_difficulty", "faculty_rating", "class_size",
    "lab_availability_hours", "internet_access", "scholarship",
    "campus_distance", "tutoring_sessions", "part_time_job_hours",
    "library_resources", "academic_advising_hours",
]

CATEGORICAL_COLS = [
    "gender", "region", "socioeconomic_status", "parent_education",
]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]
ALL_COLUMNS = [ID_COL] + FEATURE_COLS + [TARGET]

# Held-out composites used to construct the target. Excluded from predictors.
HELD_OUT_COLS = [
    "exam_performance", "engagement_index",
    "consistency_index", "wellbeing_index",
]

# Unobserved confounder (Reviewer 2: structural misspecification).
LATENT_COLS = ["A", "C", "M", "H", "U_unobserved"]

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

DEFAULT_TOTAL_ROWS = 10_000_000
DEFAULT_CHUNK_SIZE = 1_000_000