"""Transcription-factor identification from log text, dataset names and the strain DB."""

import re

import openai

from . import config, llm
from .strain_db import lookup_tfs_from_strains
from .utils import dedupe_preserve_order
from .vocabulary import TF_LOOKUP

# Newer Batgirl-format log lines look like "group: ch1_REI1 field: position".
# The TF name is the ALL-CAPS token after the underscore in the group label.
# Pattern breakdown: ch\d+_ matches the chamber prefix (ch1_, ch2_, ...),
# then ([A-Z][A-Z0-9]+) captures the uppercase TF name (e.g. REI1, MSN1).
_GROUP_TF_PATTERN = re.compile(
    r'^group\s*:\s*ch\d+_([A-Z][A-Z0-9]+)',
    re.MULTILINE,
)

# Matches "ProteinName-FluorescentTag" constructs in free-text log entries,
# e.g. "Msn2-GFP", "Dot6-mCherry2", "Whi5-YFP".
_TAGGED_PROTEIN_PATTERN = re.compile(
    r'\b([A-Za-z][A-Za-z0-9]{1,8})\s*[-–]\s*'
    r'(?:GFP2?|mCherry2?|YFP|CFP|mKO2|tdTomato|mTurquoise2?|RFP|mNeonGreen|Venus|Citrine|mKate|AID)',
    re.IGNORECASE,
)

_TF_SYSTEM_PROMPT = """\
You are extracting transcription factor (TF) names from a yeast live-cell microscopy experiment log.
Return a comma-separated list of TF names exactly as they appear in yeast nomenclature (e.g. Msn2, Mig1, Rei1).
If no TF is identifiable, return exactly: UNKNOWN
Return only the comma-separated list or UNKNOWN — no other text.
"""


def normalize_tf_name(name: str) -> str:
    """Normalise a gene name to standard yeast title-case (e.g. MSN2 → Msn2, REI1 → Rei1)."""
    if not name or name.upper() in ("UNKNOWN", ""):
        return name
    # First letter uppercase, remaining letters lowercase; digits are preserved.
    return name[0].upper() + name[1:].lower()


def parse_tf_from_dataset_name(dataset_name: str) -> list[str]:
    """Extract known TF names from an OMERO dataset name string.

    Handles common Swain-lab naming patterns:
    - CamelCase concatenation: ``Msn2Dot6Mig1``  (split at digit→uppercase junctions)
    - Underscore tokens:       ``Switch_2to0pGlc_Msn2_dIra12_00``
    - Uppercase gene tokens:   ``GAL3_MAN_GAL_00``

    Strategy: split name on ``_``, ``-``, and spaces to get coarse tokens; for each
    token attempt a direct (case-insensitive) lookup first, then re-segment by
    extracting CamelCase sub-words ([A-Z][a-z0-9]+) and look those up individually.
    """
    found = []
    for token in re.split(r'[_\-\s]', dataset_name):
        if not token:
            continue
        # Direct lookup — handles plain tokens like "Msn2", "GAL3", "Mig1"
        if token.lower() in TF_LOOKUP:
            found.append(TF_LOOKUP[token.lower()])
            continue
        # CamelCase segmentation — splits "Msn2Dot6Mig1" → ["Msn2", "Dot6", "Mig1"].
        # Pattern [A-Z][a-z0-9]+ matches one uppercase letter followed by lowercase/digits,
        # which is the standard yeast gene-name segment (Msn, Dot, Mig + trailing digits).
        for sub in re.findall(r'[A-Z][a-z0-9]+', token):
            if sub.lower() in TF_LOOKUP:
                found.append(TF_LOOKUP[sub.lower()])
    return dedupe_preserve_order(found)


def parse_tf_from_strain_text(log_text: str) -> list[str]:
    """Extract known TF names from 'Protein-FluorescentTag' constructs in log text.

    Scans for patterns like ``Msn2-GFP``, ``Dot6-mCherry``, filters by KNOWN_TFS.
    This supplements group-label and strain-DB lookups for older free-text logs.
    """
    found = []
    for m in _TAGGED_PROTEIN_PATTERN.finditer(log_text):
        protein = normalize_tf_name(m.group(1))
        canonical = TF_LOOKUP.get(protein.lower())
        if canonical:
            found.append(canonical)
    return dedupe_preserve_order(found)


def parse_tf_from_groups(
    log_text: str,
    dataset_name: str = "",
    strain_ids: list[str] | None = None,
    strain_db: dict[str, list[str]] | None = None,
    client: "openai.OpenAI | None" = None,
    model: str = config.MODEL,
) -> list[str]:
    """Identify TF names imaged in the experiment using a five-level fallback.

    Priority (highest → lowest confidence):
    1. Batgirl group labels — ``group: ch1_REI1`` → deterministic, no ambiguity.
    2. Strain DB (strain_tf_database.csv) — numeric group IDs mapped to TF names.
    3. Dataset name — extract KNOWN_TFS names from the OMERO dataset name string.
    4. Tagged-protein patterns — ``Protein-GFP`` / ``Protein-mCherry`` in log text,
       filtered against KNOWN_TFS.
    5. LLM fallback — only when all deterministic paths yield nothing.

    Returns ``["UNKNOWN"]`` when nothing is identifiable.
    """
    # Step 1: Batgirl group-label regex — normalise to title-case (REI1 → Rei1)
    raw_tfs = _GROUP_TF_PATTERN.findall(log_text)
    if raw_tfs:
        return dedupe_preserve_order(normalize_tf_name(tf) for tf in raw_tfs)

    # Step 2: strain DB lookup — authoritative source for older numeric group IDs
    if strain_ids and strain_db:
        tfs = lookup_tfs_from_strains(strain_ids, strain_db)
        if tfs:
            return tfs

    # Step 3: dataset name parsing — many names encode the TF(s) explicitly,
    # e.g. "Ramp_2to0pGlc_Msn2Dot6Mig1_00" or "GAL3_MAN_GAL_00"
    if dataset_name:
        tfs = parse_tf_from_dataset_name(dataset_name)
        if tfs:
            return tfs

    # Step 4: tagged-protein patterns in log text — "Protein-GFP" / "Protein-mCherry"
    if log_text:
        tfs = parse_tf_from_strain_text(log_text)
        if tfs:
            return tfs

    # Step 5: LLM fallback — only when all deterministic paths are exhausted
    llm_tfs = llm.complete_or_none(
        client,
        _TF_SYSTEM_PROMPT,
        log_text[:1000],
        max_tokens=60,
        model=model,
    )
    if llm_tfs and llm_tfs.upper() != "UNKNOWN":
        return dedupe_preserve_order(t.strip() for t in llm_tfs.split(",") if t.strip())

    return ["UNKNOWN"]
