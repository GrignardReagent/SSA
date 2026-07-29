"""
EXP-26-IY026: Identify fluorescence time-lapse experiments on the lab OMERO server.

Connects to the OMERO server, enumerates every accessible dataset, reads the
attached log / acquisition text annotations, and classifies each experiment as
(a) a fluorescence time-lapse and (b) a TF-localisation experiment. Regex and
database lookups do the work; the LLM is only a fallback.

All the logic lives in the ``fluorescence_survey`` package next to this file —
see README.md for what each module does. This file is only the command line.

Usage:
    /home/ianyang/micromamba/envs/alibylite/bin/python3 find_fluorescence_timelapse.py
    ... --limit 20                 # smoke-test on the first 20 datasets
    ... --resume                   # append to results.csv, skipping done IDs
    ... --output results_v2.csv    # write somewhere else

Requirements:
    - Run with the alibylite environment (it provides omero-py)
    - .env next to this file with OPENAI_API_KEY and the OMERO_* credentials
"""

import argparse
from pathlib import Path

from fluorescence_survey import config, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--output", type=Path, default=config.OUTPUT_PATH,
        help="CSV file to write results to (default: %(default)s)",
    )
    parser.add_argument(
        "--model", default=config.MODEL,
        help="OpenAI model used for the fallback classifications (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=config.LIMIT,
        help="Process at most this many datasets (default: all)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=config.RESUME,
        help="Append to an existing results CSV and skip dataset IDs already in it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline.run(
        output_path=args.output,
        model=args.model,
        limit=args.limit,
        resume=args.resume,
    )
