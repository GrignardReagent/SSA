"""Configuration for the IY026 OMERO fluorescence survey.

Every value can be overridden with an environment variable (loaded from the
``.env`` file next to the entry-point script), so no credentials or paths are
hardcoded in the analysis code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Experiment directory = parent of this package
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

# Load .env from the experiment directory before reading any env var below
load_dotenv(EXPERIMENT_DIR / ".env")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# The OMERO client libraries live in the alibylite environment, so this script
# must be run with that interpreter and alibylite's src on the import path.
ALIBYLITE_SRC = os.environ.get("ALIBYLITE_SRC", "/home/ianyang/alibylite/src")

# ---------------------------------------------------------------------------
# OMERO server
# ---------------------------------------------------------------------------
OMERO_HOST = os.environ.get("OMERO_HOST", "staffa.bio.ed.ac.uk")
OMERO_USER = os.environ.get("OMERO_USER", "upload")
OMERO_PASSWORD = os.environ.get("OMERO_PASSWORD", "")

# Ping the server this often (seconds) so long runs are not disconnected
OMERO_KEEPALIVE_SECONDS = 60

# Avoid hanging on very large OMERO file annotations. Acquisition settings are
# near the top of Acq/log files, so this still captures the useful metadata.
MAX_ANNOTATION_BYTES = 250_000

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")

# Network calls should fail loudly enough for the run to continue/resume
OPENAI_TIMEOUT_SECONDS = 30.0
OPENAI_MAX_RETRIES = 2

# Max characters of metadata text sent to the LLM (keeps token use low)
LOG_EXCERPT_CHARS = 3000

# ---------------------------------------------------------------------------
# Run behaviour
# ---------------------------------------------------------------------------
OUTPUT_PATH = EXPERIMENT_DIR / "results.csv"
# Run log; point IY026_LOG_PATH elsewhere for test runs so the committed log stays clean
LOG_PATH = Path(os.environ.get("IY026_LOG_PATH", EXPERIMENT_DIR / "find_fluorescence_timelapse.log"))
STRAIN_TF_DB_PATH = EXPERIMENT_DIR / "strain_tf_database.csv"

# Resume appends to an existing results.csv and skips dataset IDs already in it.
# Only enable when results.csv was produced by this same parser version.
RESUME = False

# Process at most this many datasets; None processes all of them
LIMIT = None

# Pause between datasets so the LLM API is not hammered (seconds)
SLEEP_BETWEEN_DATASETS = 0.2
