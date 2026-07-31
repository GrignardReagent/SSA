"""OMERO access: connect, list datasets, and pull text metadata annotations.

This is the only module that talks to the OMERO server; everything downstream
works on plain strings.
"""

import re
import sys
from dataclasses import dataclass

from . import config

# alibylite's src must be importable before the omero client is loaded
if config.ALIBYLITE_SRC not in sys.path:
    sys.path.insert(0, config.ALIBYLITE_SRC)

from omero.gateway import BlitzGateway  # noqa: E402


@dataclass
class MetadataAnnotations:
    """Text metadata attached to one OMERO dataset."""

    log_text: str | None = None
    acq_text: str | None = None
    log_filename: str = ""
    acq_filename: str = ""
    truncated_files: str = ""

    @property
    def has_metadata(self) -> bool:
        return self.log_text is not None or self.acq_text is not None

    def combined_text(self) -> str | None:
        """Return log and acquisition text concatenated, or None if neither exists."""
        parts = []
        if self.log_text:
            parts.append(f"Log file ({self.log_filename or 'unknown'}):\n{self.log_text}")
        if self.acq_text:
            parts.append(f"Acquisition file ({self.acq_filename or 'unknown'}):\n{self.acq_text}")
        return "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_omero(
    host: str = config.OMERO_HOST,
    username: str = config.OMERO_USER,
    password: str = config.OMERO_PASSWORD,
) -> BlitzGateway:
    """Open a keep-alive OMERO connection, raising if the login is rejected."""
    conn = BlitzGateway(host=host, username=username, passwd=password)
    if not conn.connect():
        raise ConnectionError(f"Could not connect to OMERO at {host}")
    conn.c.enableKeepAlive(config.OMERO_KEEPALIVE_SECONDS)
    return conn


def get_all_datasets(conn: BlitzGateway) -> list[tuple[int, str]]:
    """Return (id, name) for every dataset visible to the connected user."""
    return [(int(ds.getId()), ds.getName()) for ds in conn.getObjects("Dataset")]


# ---------------------------------------------------------------------------
# Annotation reading
# ---------------------------------------------------------------------------

# A Swain-lab microscopy log is identified by its header or its section titles
_MICROSCOPY_LOG_HEADER = "Swain Lab microscope experiment log file"
_MICROSCOPY_LOG_KEYWORDS = ("Experiment details", "Acquisition settings", "Microscope name")
# An old multiDGUI acquisition file is identified by its settings tables
_ACQ_FILE_KEYWORDS = ("Channels:", "Time_settings:", "Points:")


def _read_file_annotation_text(
    ann,
    max_bytes: int = config.MAX_ANNOTATION_BYTES,
) -> tuple[str, bool]:
    """Read up to `max_bytes` from an OMERO file annotation.

    Returns (decoded_text, was_truncated).
    """
    chunks = []
    total_bytes = 0
    truncated = False
    # Stream the annotation so a multi-MB file never lands in memory whole
    for chunk in ann.getFileInChunks():
        if total_bytes + len(chunk) > max_bytes:
            keep = max_bytes - total_bytes
            if keep > 0:
                chunks.append(chunk[:keep])
            truncated = True
            break
        chunks.append(chunk)
        total_bytes += len(chunk)
    text = b"".join(chunks).decode("utf-8", errors="ignore")
    if truncated:
        # Drop the final partial line. The cut lands mid-token, and a fragment
        # still parses: "Channel: Brightfield" cut short registered a channel
        # called "Brigh" on nine datasets.
        text = text[:text.rfind("\n") + 1] if "\n" in text else ""
    return text, truncated


# Upload bookkeeping the acquisition machine leaves in the dataset. These are
# .txt files with no microscopy content at all, and the "any other .txt" rule at
# the bottom of the priority list would otherwise accept them as metadata —
# reporting has_log=True and spending an LLM call on 31 bytes of "Upload to
# staffa is in progress" (datasets 667, 677, 974, 975).
_UPLOAD_HOUSEKEEPING_PATTERN = re.compile(
    r'upload\b.{0,40}\b(?:in progress|failed|error|completed)|'
    r'prevent upload attempts|there are no text files',
    re.IGNORECASE | re.DOTALL,
)


def _is_upload_housekeeping(text: str) -> bool:
    """True for short upload-status notes that carry no experimental metadata."""
    return len(text) < 500 and bool(_UPLOAD_HOUSEKEEPING_PATTERN.search(text))


def _is_microscopy_log(text: str) -> bool:
    return _MICROSCOPY_LOG_HEADER in text or any(k in text for k in _MICROSCOPY_LOG_KEYWORDS)


def _is_acquisition_file(fname: str, text: str) -> bool:
    return fname.endswith("acq.txt") or sum(k in text for k in _ACQ_FILE_KEYWORDS) >= 2


def read_metadata_annotations(dataset_obj) -> MetadataAnnotations:
    """
    Return decoded microscopy metadata annotations attached to the dataset.

    Old multiDGUI datasets can attach two separate files: a `*log.txt` file
    with experiment details and a `*Acq.txt` file with acquisition settings.
    Newer datasets often put the acquisition settings inside a single `.log`.

    Each candidate is scored by priority (lower = more trustworthy) and the best
    log file and the best acquisition file are kept.
    """
    metadata = MetadataAnnotations()
    log_candidates: list[tuple[int, str, str]] = []  # (priority, filename, text)
    acq_candidates: list[tuple[int, str, str]] = []

    for ann in dataset_obj.listAnnotations():
        if not hasattr(ann, "getFileName"):
            continue  # skip non-file annotations (tags, comments, etc.)
        fname = (ann.getFileName() or "").lower()
        if not (fname.endswith(".log") or fname.endswith(".txt")):
            continue  # only consider plain-text / log files

        text, truncated = _read_file_annotation_text(ann)
        if _is_upload_housekeeping(text):
            continue  # upload status note, not experimental metadata
        if truncated:
            metadata.truncated_files = ";".join(
                value for value in (metadata.truncated_files, fname) if value
            )

        # Acquisition files first: they carry the channel/time settings tables
        if _is_acquisition_file(fname, text):
            acq_candidates.append((1 if fname.endswith("acq.txt") else 2, fname, text))
            continue
        if _is_microscopy_log(text):
            log_candidates.append((1, fname, text))
            continue
        if fname.endswith("log.txt"):
            log_candidates.append((2, fname, text))   # older multiDGUI naming convention
        elif fname.endswith(".log"):
            log_candidates.append((3, fname, text))   # newer Batgirl format
        else:
            log_candidates.append((4, fname, text))   # last resort: any .txt file

    if log_candidates:
        _, metadata.log_filename, metadata.log_text = sorted(log_candidates, key=lambda x: x[0])[0]
    if acq_candidates:
        _, metadata.acq_filename, metadata.acq_text = sorted(acq_candidates, key=lambda x: x[0])[0]
    return metadata
