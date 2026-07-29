"""Run loop: walk every OMERO dataset, classify it, and stream rows to results.csv."""

import csv
import logging
import sys
import time
from pathlib import Path

from . import config, llm
from .classification import RESULT_FIELDS, classify, error_result
from .omero_source import connect_omero, get_all_datasets, read_metadata_annotations
from .strain_db import load_strain_tf_database


def setup_logging(log_path: Path = config.LOG_PATH) -> logging.Logger:
    """Log to both stdout and the run log file, muting noisy third-party loggers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
        force=True,
    )
    for noisy_logger in ("httpcore", "httpx", "openai", "omero"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    return logging.getLogger("fluorescence_survey")


def load_already_processed(path: Path) -> set[int]:
    """Return set of dataset IDs already written to the output CSV."""
    if not path.exists():
        return set()
    dataset_ids = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                dataset_ids.add(int(row["dataset_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return dataset_ids


def run(
    output_path: Path = config.OUTPUT_PATH,
    model: str = config.MODEL,
    limit: int | None = config.LIMIT,
    resume: bool = config.RESUME,
) -> None:
    """Classify every visible OMERO dataset and write results to `output_path`."""
    log = setup_logging()

    if not config.OPENAI_API_KEY:
        log.error("No OpenAI API key found. Set OPENAI_API_KEY in the .env file.")
        sys.exit(1)

    # Resume skips dataset IDs already present in the output CSV
    already_done = load_already_processed(output_path) if resume else set()
    if already_done:
        log.info(f"Resuming: {len(already_done)} datasets already processed.")

    strain_db = load_strain_tf_database()
    if strain_db:
        log.info(f"Loaded strain TF database: {len(strain_db)} entries from {config.STRAIN_TF_DB_PATH}.")
    else:
        log.warning(
            f"Strain TF database not found at {config.STRAIN_TF_DB_PATH}; "
            "TF lookup from strain ID disabled."
        )

    log.info(f"Connecting to OMERO at {config.OMERO_HOST}...")
    conn = connect_omero()
    log.info("Connected.")

    client = llm.make_client()

    try:
        all_datasets = get_all_datasets(conn)
        log.info(f"Found {len(all_datasets)} datasets on the server.")

        datasets_to_process = [
            (ds_id, ds_name)
            for ds_id, ds_name in all_datasets
            if ds_id not in already_done
        ]
        if limit:
            datasets_to_process = datasets_to_process[:limit]
        log.info(f"Will process {len(datasets_to_process)} datasets.")

        # Resume appends to existing results; fresh runs overwrite to avoid duplicate headers.
        write_header = not output_path.exists() or output_path.stat().st_size == 0 or not resume
        with open(output_path, "a" if resume else "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
            if write_header:
                writer.writeheader()

            # --- main loop: one OMERO dataset per iteration ---
            for i, (ds_id, ds_name) in enumerate(datasets_to_process, 1):
                log.info(f"[{i}/{len(datasets_to_process)}] Dataset {ds_id}: {ds_name}")
                try:
                    ds_obj = conn.getObject("Dataset", ds_id)
                    log.info("  Reading metadata annotations...")
                    metadata = read_metadata_annotations(ds_obj)
                    if not metadata.has_metadata:
                        log.warning(f"  No .log/.txt/Acq.txt annotation found for dataset {ds_id}.")
                    elif metadata.truncated_files:
                        log.warning(f"  Truncated large annotation(s): {metadata.truncated_files}")

                    log.info("  Classifying metadata...")
                    result = classify(client, ds_id, ds_name, metadata, model=model, strain_db=strain_db)
                except Exception as exc:
                    # One bad dataset must not end the run: record it and continue
                    log.exception(f"  Failed dataset {ds_id}: {exc}")
                    result = error_result(ds_id, ds_name, exc)

                writer.writerow(result)
                csv_file.flush()  # write through after each row so results survive a crash mid-run

                log.info(
                    f"  -> fluor: {result['classification']} "
                    f"| tf_loc: {result['is_tf_localisation']} "
                    f"| tps: {result['timepoints']} "
                    f"| tf: {result['tf_identity']} "
                    f"| condition: {result['condition']} "
                    f"| channels: {result['channels']}"
                )

                time.sleep(config.SLEEP_BETWEEN_DATASETS)  # polite pause for the LLM API
    finally:
        conn.close()

    # --- summary over the full output file ---
    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))
    yes_count = sum(1 for r in rows if r["classification"] == "YES")
    log.info(f"Finished. {yes_count}/{len(rows)} experiments classified as fluorescence time-lapse.")
    log.info(f"Results saved to {output_path}")
