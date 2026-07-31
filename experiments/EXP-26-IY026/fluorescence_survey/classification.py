"""The two classifications produced for every dataset.

1. ``classification``        — is this a fluorescence TIME-LAPSE experiment?
                               (>=1 fluorescence channel AND >1 timepoint)
2. ``is_tf_localisation``    — is a transcription factor being tracked?

Both prefer deterministic evidence and only call the LLM when the parsers are
inconclusive.
"""

from typing import TYPE_CHECKING

from . import config, llm
from .omero_source import MetadataAnnotations
from .parse_channels import fluorescence_channels, parse_channels
from .parse_experiment import parse_condition, parse_strain_ids, parse_timepoints
from .parse_tf import parse_tf_from_groups
from .strain_db import lookup_tfs_from_strains
from .utils import Parsed
from .vocabulary import BRIGHTFIELD_CHANNELS, FLUORESCENCE_CHANNELS, NON_TF_LOOKUP, TF_LOOKUP

if TYPE_CHECKING:
    import openai

# Column order of results.csv
RESULT_FIELDS = [
    "dataset_id",
    "dataset_name",
    "has_log",
    "condition",
    "strain_id",
    "tf_identity",
    "timepoints",
    "classification",        # fluorescence time-lapse (YES/NO/ERROR)
    "is_tf_localisation",    # TF localisation experiment (YES/NO/UNKNOWN)
    "tf_localisation_reason",
    "reason",
    "channels",              # fluorescence channels only
    "all_channels",          # every channel acquired, brightfield included
    "provenance",            # which extractor produced each field
    "raw_llm_response",      # every LLM reply, labelled by call
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The LLM is only asked about fluorescence — timepoints and strain are extracted by
# code. The channel list is built from the vocabulary rather than written out, so
# the prompt cannot drift out of step with what the parser recognises.
FLUORESCENCE_SYSTEM_PROMPT = f"""\
You are a lab data curator analysing microscopy experiment metadata from the
Swain lab (yeast live-cell imaging). You receive a dataset name and the first
~3000 characters of a log file (which may be in a newer Swainlab format or an
older multiDGUI format) and must determine whether fluorescence imaging was used.

Criterion: at least one fluorescence channel was acquired. Fluorescence channels
include: {', '.join(FLUORESCENCE_CHANNELS)}.
Transmitted light alone ({', '.join(BRIGHTFIELD_CHANNELS)}) is NOT fluorescence.

Respond ONLY in this exact format (three lines, no extra text):
FLUORESCENCE: YES or NO
REASON: one sentence
CHANNELS: comma-separated channel names found, or UNKNOWN
"""

FLUORESCENCE_USER_TEMPLATE = """\
Dataset ID: {dataset_id}
Dataset name: {dataset_name}

Metadata excerpt:
---
{log_excerpt}
---
"""

TF_LOCALISATION_SYSTEM_PROMPT = """\
You are a lab data curator for Swain-lab yeast live-cell microscopy experiments.
Determine whether this experiment is a TRANSCRIPTION FACTOR (TF) LOCALISATION experiment.

A TF localisation experiment:
- Tags one or more transcription factors (or nuclear-localisation reporters) with a fluorescent protein (GFP, mCherry, …)
- Monitors the nuclear-vs-cytoplasmic distribution of the TF over time in response to a stimulus
- Examples of TFs: Msn2, Msn4, Mig1, Dot6, Sfp1, Hog1, Crz1, Yap1, Whi5, Opi1, Swi4, etc.

NOT a TF localisation experiment:
- Organelle markers only (Vph1 for vacuole, Cox4 for mitochondria, Pex14 for peroxisome)
- Histone markers (Htb2-GFP, Hta2-mCherry) — these mark the nucleus but are not TFs
- Stress-granule / P-body markers (Pab1-GFP, Edc1-GFP)
- pH or metabolite sensors (pHluorin, NADH)
- Morphology / bud-neck markers (Bud3, Lte1)
- Protein aggregation screens (non-TF metabolic enzymes)
- Pure brightfield or structural imaging

Respond ONLY in this exact format (two lines, no extra text):
TF_LOCALISATION: YES or NO
REASON: one sentence — name the TF(s) being tracked, or explain why this is not TF localisation
"""


# ---------------------------------------------------------------------------
# TF localisation
# ---------------------------------------------------------------------------

def classify_tf_localisation(
    tf_names: list[str],
    dataset_name: str,
    metadata_text: str | None,
    strain_ids: list[str],
    strain_db: dict[str, list[str]],
    client: "llm.Session | openai.OpenAI | None",
    model: str = config.MODEL,
) -> Parsed[tuple[str, str]]:
    """Determine whether the experiment tracks TF localisation.

    Returns (is_tf_loc, reason) where is_tf_loc is 'YES', 'NO', or 'UNKNOWN'.

    Priority:
    1. Strain IDs in strain_tf_database → YES (highest confidence).
    2. At least one detected protein is in KNOWN_TFS → YES.
    3. Detected proteins are all in KNOWN_NON_TF_MARKERS → NO.
    4. LLM with a targeted prompt — only when deterministic paths are inconclusive.
    """
    real_tfs = [tf for tf in tf_names if tf.lower() not in ("unknown", "")]

    # Step 1: strain DB — authoritative; these group IDs were specifically mapped to TFs
    db_tfs = lookup_tfs_from_strains(strain_ids, strain_db)
    if db_tfs:
        reason = f"Strain(s) {'; '.join(strain_ids[:4])} in TF database → {', '.join(db_tfs)}"
        return Parsed(("YES", reason), ("strain-db",))

    if real_tfs:
        known_tf_hits = [tf for tf in real_tfs if tf.lower() in TF_LOOKUP]
        non_tf_hits = [tf for tf in real_tfs if tf.lower() in NON_TF_LOOKUP]

        # Step 2: at least one protein is a confirmed TF → YES (TF channel(s) present)
        if known_tf_hits:
            return Parsed(
                ("YES", f"Known TF(s) detected: {', '.join(known_tf_hits)}"),
                ("known-tf",),
            )

        # Step 3: all detected proteins are non-TF markers → NO
        if non_tf_hits and len(non_tf_hits) == len(real_tfs):
            return Parsed(
                ("NO", f"Only non-TF markers detected: {', '.join(non_tf_hits)}"),
                ("non-tf-marker",),
            )

    # Step 4: LLM fallback
    if metadata_text or dataset_name:
        excerpt = (metadata_text or "")[:1500]
        llm_text = llm.complete_or_none(
            client,
            TF_LOCALISATION_SYSTEM_PROMPT,
            f"Dataset name: {dataset_name}\n\n{excerpt}",
            max_tokens=80,
            model=model,
            label="tf-localisation",
        )
        if llm_text is not None:
            fields = llm.parse_labelled_lines(llm_text, ("TF_LOCALISATION", "REASON"))
            return Parsed(
                (fields["TF_LOCALISATION"] or "UNKNOWN", fields["REASON"]),
                ("llm",),
            )

    return Parsed(("UNKNOWN", "Insufficient metadata to determine TF localisation"), ())


# ---------------------------------------------------------------------------
# Fluorescence time-lapse
# ---------------------------------------------------------------------------

def build_metadata_excerpt(metadata: MetadataAnnotations) -> str:
    """Return a balanced LLM excerpt from log and acquisition files."""
    if not metadata.has_metadata:
        return "(no log or acquisition file attached)"
    # When both files exist, give each half of the character budget
    if metadata.log_text and metadata.acq_text:
        half = config.LOG_EXCERPT_CHARS // 2
        return (
            f"Log file ({metadata.log_filename}):\n{metadata.log_text[:half]}\n\n"
            f"Acquisition file ({metadata.acq_filename}):\n{metadata.acq_text[:half]}"
        )
    combined = metadata.combined_text()
    return combined[:config.LOG_EXCERPT_CHARS] if combined else "(no log or acquisition file attached)"


def classify(
    client: "llm.Session | openai.OpenAI | None",
    dataset_id: int,
    dataset_name: str,
    metadata: MetadataAnnotations,
    model: str = config.MODEL,
    strain_db: dict[str, list[str]] | None = None,
) -> dict:
    """Build the complete results.csv row for one dataset."""
    # Every LLM call this dataset makes goes through the session, which keeps the
    # replies for the raw_llm_response audit column
    session = llm.Session(client)

    # --- deterministic extraction from log text ---
    metadata_text = metadata.combined_text()
    condition = (
        parse_condition(metadata_text, client=session, model=model)
        if metadata_text else Parsed("", ())
    )
    timepoints = parse_timepoints(metadata_text) if metadata_text else None
    strain_ids = parse_strain_ids(metadata_text) if metadata_text else Parsed([], ())

    # TF identification: six-level fallback (group labels → strain DB → merged
    # free-text sources → LLM). LLM is last resort only.
    tf_identity = parse_tf_from_groups(
        metadata_text or "",
        dataset_name=dataset_name,
        strain_ids=strain_ids.value,
        strain_db=strain_db,
        client=session,
        model=model,
    )

    parsed_channels = parse_channels(metadata_text) if metadata_text else []
    parsed_fluorescence_channels = fluorescence_channels(parsed_channels)

    # --- fluorescence verdict: three mutually exclusive paths ---
    if not metadata.has_metadata:
        # Nothing to read → cannot have been fluorescence
        fluorescence_source = "no-metadata"
        llm_text = (
            "FLUORESCENCE: NO\n"
            "REASON: No log or acquisition metadata was attached.\n"
            "CHANNELS: UNKNOWN"
        )
    elif parsed_fluorescence_channels:
        # Deterministic hit → no LLM call needed
        fluorescence_source = "parsed-channels"
        llm_text = (
            "FLUORESCENCE: YES\n"
            "REASON: Fluorescence channel(s) found in parsed acquisition metadata.\n"
            f"CHANNELS: {', '.join(parsed_fluorescence_channels)}"
        )
    elif session.client is None:
        # No model configured: report what the parser saw rather than crashing, so
        # the deterministic pipeline can be run and tested on its own
        fluorescence_source = "no-model"
        llm_text = (
            "FLUORESCENCE: NO\n"
            "REASON: No fluorescence channel found in the parsed acquisition metadata.\n"
            "CHANNELS: UNKNOWN"
        )
    else:
        # Metadata exists but no channel was recognised → ask the LLM
        fluorescence_source = "llm"
        user_msg = FLUORESCENCE_USER_TEMPLATE.format(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            log_excerpt=build_metadata_excerpt(metadata),
        )
        llm_text = llm.complete(
            session,
            FLUORESCENCE_SYSTEM_PROMPT,
            user_msg,
            max_tokens=150,
            model=model,
            label="fluorescence",
        )

    # Parse the three structured lines from the fluorescence response
    fields = llm.parse_labelled_lines(llm_text, ("FLUORESCENCE", "REASON", "CHANNELS"))
    fluorescence = fields["FLUORESCENCE"] or "UNKNOWN"
    reason = fields["REASON"]
    channels = fields["CHANNELS"]

    # Fluorescence time-lapse: YES only if fluorescence detected AND timepoints > 1
    has_fluorescence = fluorescence == "YES" or bool(parsed_fluorescence_channels)
    is_timelapse = timepoints is not None and timepoints > 1
    classification = "YES" if (has_fluorescence and is_timelapse) else "NO"
    if parsed_fluorescence_channels:
        channels = ";".join(parsed_fluorescence_channels)
    if not reason and parsed_fluorescence_channels:
        reason = "Fluorescence channel(s) found in parsed acquisition metadata."

    # TF localisation classification: separate from fluorescence classification.
    # Only runs deterministic + LLM check; does not require a second OMERO call.
    tf_localisation = classify_tf_localisation(
        tf_names=tf_identity.value,
        dataset_name=dataset_name,
        metadata_text=metadata_text,
        strain_ids=strain_ids.value,
        strain_db=strain_db or {},
        client=session,
        model=model,
    )
    is_tf_loc, tf_loc_reason = tf_localisation.value

    # How far to trust each field: which link of each fallback chain produced it
    provenance = " | ".join([
        f"condition={condition.label()}",
        f"strain={strain_ids.label()}",
        f"tf={tf_identity.label()}",
        f"fluorescence={fluorescence_source}",
        f"tf_localisation={tf_localisation.label()}",
    ])

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "has_log": metadata.has_metadata,
        "condition": condition.value,
        "strain_id": ";".join(strain_ids.value),
        "tf_identity": ";".join(tf_identity.value),
        "timepoints": timepoints if timepoints is not None else "",
        "classification": classification,
        "is_tf_localisation": is_tf_loc,
        "tf_localisation_reason": tf_loc_reason,
        "reason": (
            reason if not metadata.truncated_files
            else f"{reason} Metadata truncated: {metadata.truncated_files}."
        ),
        "channels": channels,
        "all_channels": ";".join(parsed_channels),
        "provenance": provenance,
        "raw_llm_response": session.transcript(),
    }


def error_result(dataset_id: int, dataset_name: str, exc: Exception) -> dict:
    """Return a CSV row for a dataset that failed without stopping the run."""
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "has_log": False,
        "condition": "",
        "strain_id": "",
        "tf_identity": "",
        "timepoints": "",
        "classification": "ERROR",
        "is_tf_localisation": "UNKNOWN",
        "tf_localisation_reason": "",
        "reason": f"{type(exc).__name__}: {exc}",
        "channels": "",
        "all_channels": "",
        "provenance": "",
        "raw_llm_response": "",
    }
