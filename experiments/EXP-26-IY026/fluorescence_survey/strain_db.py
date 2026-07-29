"""Strain ID → transcription factor lookup, backed by strain_tf_database.csv.

The CSV is the manually curated authority for older experiments whose logs only
record a numeric group ID (e.g. ``group: 898``) rather than a TF name.
"""

import csv
from pathlib import Path

from . import config
from .utils import dedupe_preserve_order


def load_strain_tf_database(db_path: Path = config.STRAIN_TF_DB_PATH) -> dict[str, list[str]]:
    """Load strain_tf_database.csv and return {group_id: [tf, ...]} mapping.

    The CSV has columns: group_id, channel, tf. A single group_id can have
    multiple rows (one per channel, e.g. GFP and mCherry), so each key maps
    to a list of TF names rather than a single string.
    """
    db: dict[str, list[str]] = {}
    if not db_path.exists():
        return db
    with open(db_path, newline="") as f:
        for row in csv.DictReader(f):
            gid = row.get("group_id", "").strip()
            tf = row.get("tf", "").strip()
            if gid and tf:
                db.setdefault(gid, [])
                if tf not in db[gid]:  # avoid duplicates when multiple channels share a TF
                    db[gid].append(tf)
    return db


def lookup_tfs_from_strains(strain_ids: list[str], db: dict[str, list[str]]) -> list[str]:
    """Return TF names for the given strain/group IDs from the database.

    Used for older experiments where the group ID is a plain number (e.g. 898)
    rather than an explicit TF name in the group label.
    """
    tfs = []
    for sid in strain_ids:
        tfs.extend(db.get(sid, []))  # returns [] if strain not in db
    return dedupe_preserve_order(tfs)
