"""Transcription-factor identification from log text, dataset names and the strain DB."""

import re
from typing import TYPE_CHECKING

from . import config, llm
from .parse_experiment import parse_experiment_details, parse_position_names, strain_description
from .strain_db import lookup_tfs_from_strains
from .utils import Parsed, dedupe_preserve_order
from .vocabulary import NON_TF_LOOKUP, TF_LOOKUP

if TYPE_CHECKING:  # the parsers here are pure text-in/values-out; only the
    import openai  # optional LLM fallback needs the SDK at run time

# Group labels are matched against every catalogued protein, TFs and non-TF
# markers alike: "group: ch1_VPH1" must still yield Vph1 so that
# `classify_tf_localisation` can rule the experiment out as a vacuole marker.
_PROTEIN_LOOKUP = {**NON_TF_LOOKUP, **TF_LOOKUP}

# Batgirl-format log lines look like "group: ch1_REI1 field: position"; this
# grabs the whole label so it can be inspected below.
_GROUP_LABEL_PATTERN = re.compile(r'(?im)^group\s*:\s*(\S+)')

# A label made only of a gene name and its position on the microscope: a chamber
# number either side of the gene ("ch1_REI1", "Aft1_ch18"), optionally with the
# source plate well between them ("ch10_C11_RPS18A", "ch17_d7_EDC1"). Everything
# except the gene is positional, so the gene needs no vocabulary lookup — which is
# what recovers screens naming genes nobody has catalogued: dataset 926 images 19
# TF chambers, six of them absent from KNOWN_TFS.
#
# The well must be distinguished from the gene, or the wrong half is read: the
# plate screens (datasets 1666-2462) used to report `C11` and `A1` as TF names. A
# well is one letter and a number; a gene name is longer.
_CHAMBER_LABEL = re.compile(
    r'^(?:ch\d+_(?:[A-Za-z]\d{1,2}_)?([A-Za-z][A-Za-z0-9]+)|([A-Za-z][A-Za-z0-9]+)_ch\d+)$',
    re.IGNORECASE,
)

# Matches "ProteinName-FluorescentTag" constructs in free-text log entries,
# e.g. "Msn2-GFP", "Dot6-mCherry2", "Whi5-YFP", "htb2::mCherry" (the standard
# yeast allele notation for a tagged locus).
_TAGGED_PROTEIN_PATTERN = re.compile(
    r'\b([A-Za-z][A-Za-z0-9]{1,8})\s*(?:[-–]|::)\s*'
    r'(?:sfGFP|GFP2?|mCherry2?|YFP|CFP|mKO2|tdTomato|mTurquoise2?|RFP|mNeonGreen|Venus|Citrine|mKate|AID)',
    re.IGNORECASE,
)

# Any run of non-alphanumeric characters separates one name from the next in a
# strain description: "Msn2-GFP/Dot6-mCherry", "87 (Msn2-GFP), 416 (Hog1)".
_NAME_SEPARATOR_PATTERN = re.compile(r'[^A-Za-z0-9]+')

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


def _proteins_in_labels(labels: list[str]) -> list[str]:
    """Read protein names out of chamber labels, in the four shapes the lab writes.

    - ``ch1_REI1``   gene and chamber  → the half that is not the chamber wins
    - ``ACE2_ch17``  same, reversed    → likewise
    - ``Msn2_GFP``   gene plus its tag → vocabulary lookup
    - ``Dot6``       bare gene name    → vocabulary lookup

    The first two need no vocabulary, because a label made of a gene name and a
    chamber number holds nothing else. The last two do: without it ``GFP`` in
    ``Msn2_GFP`` would be read as a protein alongside Msn2.
    """
    found = []
    for label in labels:
        chamber_label = _CHAMBER_LABEL.match(label)
        if chamber_label:
            found.append(normalize_tf_name(chamber_label.group(1) or chamber_label.group(2)))
            continue
        for token in _NAME_SEPARATOR_PATTERN.split(label):
            protein = _PROTEIN_LOOKUP.get(token.lower())
            if protein:
                found.append(protein)
    return dedupe_preserve_order(found)


def parse_tf_from_group_labels(log_text: str) -> list[str]:
    """Extract protein names from Batgirl ``group:`` labels.

    The acquisition software writes one label per imaged chamber, naming the
    strain in it, which makes this the most reliable TF evidence in the log.

    Reading only ``ch1_REI1``, and only in upper case, left this source resolving
    5 of the 70 datasets that have group labels in a 220-dataset sample; all four
    shapes together resolve 18.
    """
    return _proteins_in_labels(_GROUP_LABEL_PATTERN.findall(log_text))


def parse_tf_from_position_names(log_text: str) -> list[str]:
    """Extract protein names from the names given to each imaged position.

    The fallback for experiments whose ``Strain:`` field says only "see position
    names" (dataset 824: ``Dot6``, ``Hog1``, ``Msn2``…). Unlike ``group:`` labels
    these are typed by hand and need not name the fluorescent reporter: dataset
    823's positions are named after the AID-tagged degradation targets while the
    imaged reporters, Mig1-mCherry and Msn2-mCherry, appear only in the ``Strain:``
    field. So this is merged with the other free-text sources rather than trusted
    on its own.
    """
    return _proteins_in_labels(parse_position_names(log_text))


def parse_tf_from_details(details: str) -> list[str]:
    """Extract known TF names from the strain description in the Experiment details.

    Reads the ``Strain:`` field and any prose sentence about strains, and looks up
    every name-like token against ``KNOWN_TFS``. Unlike
    :func:`parse_tf_from_strain_text` this does not require a fluorescent tag, so
    it also catches the many logs that list the TFs bare:

    - ``Strain: Mig1, Dot6, Maf1, Sfp1, Msn2``
    - ``Strain: 87 (Msn2-GFP), 416 (Hog1), 424 (Dot6), 429(Yap1)``
    - ``Strains are 87 (msn2-GFP) and 1138 (msn2-mCherry)``

    Gating on the vocabulary is what keeps this safe: strain descriptions are full
    of plasmid names, background strains and free text, and none of them can leak
    in unless they are a confirmed TF.
    """
    if not details:
        return []
    field, sentences = strain_description(details)
    segments = [field] if field else sentences

    found = []
    for segment in segments:
        for token in _NAME_SEPARATOR_PATTERN.split(segment):
            canonical = TF_LOOKUP.get(token.lower())
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
) -> Parsed[list[str]]:
    """Identify every TF imaged in the experiment.

    The two enumerating sources short-circuit, because each already lists one TF
    per imaged group and anything else would only add noise:

    1. Batgirl group labels — ``ch1_REI1``, ``ACE2_ch17``, ``Msn2_GFP``, ``Dot6``.
    2. Strain DB (strain_tf_database.csv) — numeric group IDs mapped to TF names.

    Failing those, the three free-text sources are *merged* rather than ranked,
    because each is only a partial view of the same strain list and none is
    reliably complete:

    3. Experiment details — the ``Strain:`` field or a prose strain sentence.
    4. Position names — for experiments whose ``Strain:`` field says only
       "see position names".
    5. Dataset name — ``Ramp_2to0pGlc_Msn2Dot6Mig1_00``, ``GAL3_MAN_GAL_00``.
    6. Tagged-protein constructs anywhere in the log — ``Msn2-GFP``, and notably
       the OMERO tag line, which often records a strain the ``Strain:`` field
       abbreviated or mistyped.

    These are filtered against the protein vocabulary, so merging cannot let an
    arbitrary gene name in. Only when they are all empty is the LLM asked (7), and
    ``["UNKNOWN"]`` returned if it too finds nothing.
    """
    # Step 1: group labels — the strain imaged in each chamber, written by the
    # acquisition software rather than by hand
    tfs = parse_tf_from_group_labels(log_text)
    if tfs:
        return Parsed(tfs, ("group-labels",))

    # Step 2: strain DB lookup — authoritative source for older numeric group IDs
    if strain_ids and strain_db:
        tfs = lookup_tfs_from_strains(strain_ids, strain_db)
        if tfs:
            return Parsed(tfs, ("strain-db",))

    # Steps 3-6: merged free-text sources, most specific first so that the
    # experimenter's own strain list leads the column
    contributions = [
        ("details", parse_tf_from_details(parse_experiment_details(log_text)) if log_text else []),
        ("position-names", parse_tf_from_position_names(log_text) if log_text else []),
        ("dataset-name", parse_tf_from_dataset_name(dataset_name) if dataset_name else []),
        ("tagged-proteins", parse_tf_from_strain_text(log_text) if log_text else []),
    ]
    tfs = [tf for _, found in contributions for tf in found]
    if tfs:
        sources = tuple(name for name, found in contributions if found)
        return Parsed(dedupe_preserve_order(tfs), sources)

    # Step 6: LLM fallback — only when all deterministic paths are exhausted
    llm_tfs = llm.complete_or_none(
        client,
        _TF_SYSTEM_PROMPT,
        log_text[:1000],
        max_tokens=60,
        model=model,
        label="tf-identity",
    )
    if llm_tfs and llm_tfs.upper() != "UNKNOWN":
        # Title-cased like every other source, or the column mixes "Gal1" and "GAL1"
        return Parsed(
            dedupe_preserve_order(
                normalize_tf_name(tf.strip()) for tf in llm_tfs.split(",") if tf.strip()
            ),
            ("llm",),
        )

    return Parsed(["UNKNOWN"], ())
