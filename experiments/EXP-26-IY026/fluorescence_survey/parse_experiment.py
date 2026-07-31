"""Experiment-level fields parsed from log text: timepoints, strains, condition."""

import csv
import re
from typing import TYPE_CHECKING

from . import config, llm
from .utils import Parsed, dedupe_preserve_order

if TYPE_CHECKING:  # the parsers here are pure text-in/values-out; only the
    import openai  # optional LLM fallback needs the SDK at run time

# ---------------------------------------------------------------------------
# Timepoints
# ---------------------------------------------------------------------------

# Old multiDGUI acquisition table: the 3rd column of the Time_settings row is
# the number of timepoints.
_ACQ_TIME_SETTINGS_PATTERN = re.compile(
    r'(?im)^Time_settings:\s*\n\s*[^,\n]+,\s*[^,\n]+,\s*(\d+)\s*,'
)

# Explicit declarations of the total, ordered by specificity; first match wins
_TIMEPOINTS_PATTERNS = [
    re.compile(r'number\s+of\s+timepoints\s*[=:]\s*(\d+)', re.IGNORECASE),
    re.compile(r'ntimepoints\s*[=:]\s*(\d+)', re.IGNORECASE),
    re.compile(r'^frames\s*:\s*(\d+)', re.IGNORECASE | re.MULTILINE),
    re.compile(r'\btime\s*point\s+\d+\s*(?:/|of)\s*(\d+)', re.IGNORECASE),
    re.compile(r'\btimepoint\s+\d+\s*(?:/|of)\s*(\d+)', re.IGNORECASE),
    re.compile(r'\bframes?\s+\d+\s*(?:/|of)\s*(\d+)', re.IGNORECASE),
]

# Last resort: per-timepoint progress lines ("--- Time point 137 ---"); the
# highest index seen is the number of timepoints actually acquired.
_TIMEPOINT_INDEX_PATTERN = re.compile(
    r'^\s*-*\s*time\s*points?[_\s:=]*(\d+)\b',
    re.IGNORECASE | re.MULTILINE,
)


def parse_timepoints(log_text: str) -> int | None:
    """Return the number of acquired timepoints parsed from the log, or None if not found."""
    # 1. Acquisition settings table (most reliable)
    m = _ACQ_TIME_SETTINGS_PATTERN.search(log_text)
    if m:
        return int(m.group(1))
    # 2. Explicit "number of timepoints = N" style declarations
    for pattern in _TIMEPOINTS_PATTERNS:
        m = pattern.search(log_text)
        if m:
            return int(m.group(1))
    # 3. Count from progress lines
    indexed_timepoints = [int(m.group(1)) for m in _TIMEPOINT_INDEX_PATTERN.finditer(log_text)]
    if indexed_timepoints:
        return max(indexed_timepoints)
    return None


# ---------------------------------------------------------------------------
# The "Experiment details" block
# ---------------------------------------------------------------------------

# The "Experiment details:" line runs until the next section separator ("---...---")
# or the start of "Acquisition settings". re.DOTALL lets . match newlines so
# multi-line details are captured in one match.
_EXP_DETAILS_PATTERN = re.compile(
    r'Experiment details:\s*(.+?)(?=\n\s*-{3,}|\n\s*Acquisition settings|\Z)',
    re.IGNORECASE | re.DOTALL,
)

# Headings that begin the machine-generated section of the log. Everything from
# the first of them onwards is acquisition software output, not the experimenter's
# description: config filenames, filter-set numbers and OMERO bookkeeping, all of
# which would otherwise be mistaken for experimental values. Older logs skip
# "Microscope setup" and go straight to "Omero project:", so all are listed.
_DETAILS_BOILERPLATE_PATTERN = re.compile(
    r'\n?\s*(?:Microscope setup for used channels|Micromanager config file|'
    r'Omero project|Omero tags|Omero tag descriptions|Experiment started at)\s*:.*',
    re.IGNORECASE | re.DOTALL,
)

# Experimenters sometimes type the field label into the field, giving
# "Experiment details: Experiment details: ..." (datasets 2806, 2808, 2810).
_REPEATED_DETAILS_LABEL_PATTERN = re.compile(r'^\s*Experiment details:\s*', re.IGNORECASE)


def parse_experiment_details(log_text: str) -> str:
    """Return the experimenter-written 'Experiment details' text, or '' if absent.

    This is the free-text block at the top of a Swain-lab log holding the ``Aim:``,
    ``Strain:`` and ``Comments:`` fields (newer logs) or a plain prose description
    (older ones). The machine-generated configuration that follows it is stripped,
    as is a field label the experimenter repeated by hand.
    """
    m = _EXP_DETAILS_PATTERN.search(log_text)
    if not m:
        return ""
    details = _DETAILS_BOILERPLATE_PATTERN.sub("", m.group(1))
    return _REPEATED_DETAILS_LABEL_PATTERN.sub("", details).strip()


# ---------------------------------------------------------------------------
# Strain / group IDs
# ---------------------------------------------------------------------------

# Batgirl-format position lines: "group: 898 field: ..."
_STRAIN_PATTERN = re.compile(r'^group\s*:\s*(\d+)', re.IGNORECASE | re.MULTILINE)

# Field labels that end the free-text "Strain:" value. Without them the value
# runs on into the rest of the log — logs are inconsistently spaced, so e.g.
# "Strain:247Comments: Omero tags: 13-Sep-2017,..." must still stop at 247.
_DETAILS_FIELD_LABELS = (
    "Comments", "Comment", "Notes", "Note", "Aim",
    "Omero project", "Omero tags", "Omero tag descriptions",
    "Micromanager config file", "Experiment started at",
)

# Free-text "Strain:" field inside the Experiment details block. The value may
# span several lines (one strain per line), so it is terminated by the next
# known field label rather than by a newline.
_STRAIN_FIELD_PATTERN = re.compile(
    r'\bStrains?\s*:\s*(.*?)'
    r'(?=(?:' + "|".join(re.escape(label) for label in _DETAILS_FIELD_LABELS) + r')\s*:|\Z)',
    re.IGNORECASE | re.DOTALL,
)

# Strains are referenced in log text as "YST_1490", "YST1490", "YST-708" or
# "yst365" (lab naming convention). (\d+) captures just the numeric part.
_YST_STRAIN_PATTERN = re.compile(r'\bYST[\s_-]?(\d+)\b', re.IGNORECASE)

# A bare lab strain number: 2-4 digits with no leading zero, not glued to a
# letter, digit, underscore or hyphen. The guards reject the many non-strain
# numbers that share the details block — background-strain names ("BY4741",
# "W303", "CBS138"), position ranges ("pos001-006") and codes ("5_75").
_BARE_STRAIN_ID_PATTERN = re.compile(r'(?<![A-Za-z0-9_-])([1-9]\d{1,3})(?![A-Za-z0-9_])')

# The same number, but only where a parenthetical description follows it, as in
# "87 (msn2-GFP)". This is the one form trusted outside the "Strain:" field,
# where an unqualified number is far more likely to be a time or a concentration.
_ANNOTATED_STRAIN_ID_PATTERN = re.compile(r'(?<![A-Za-z0-9_-])([1-9]\d{1,3})\s*\(')

# A parenthetical glued straight onto a gene name is a residue range, not a
# strain ID: "Msn2(604-636)", "Msn4(316)". Dropped before scanning for numbers.
# "429(Yap1)" is left alone because the token before the bracket is numeric.
_GENE_PARENTHETICAL_PATTERN = re.compile(r'\b[A-Za-z][A-Za-z0-9]*\([^)]*\)')

# Sentence splitter for prose-style details blocks, and the cue word that marks
# a sentence as being about strains.
_SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
_STRAIN_WORD_PATTERN = re.compile(r'\bstrains?\b', re.IGNORECASE)


def parse_strain_field(details: str) -> str:
    """Return the raw value of the ``Strain:`` field in the details block, or ''."""
    m = _STRAIN_FIELD_PATTERN.search(details)
    return m.group(1).strip() if m else ""


def strain_description(details: str) -> tuple[str, list[str]]:
    """Return the two places a details block describes its strains.

    ``(strain_field_value, prose_sentences)``, where the prose is every sentence
    mentioning a strain, e.g. "Strains are 87 (msn2-GFP) and 1138 (msn2-mCherry)."

    The prose is only searched when the ``Strain:`` field is absent or blank. Logs
    that have the field have an authoritative list already, and their fields and
    comments often run together into one unpunctuated "sentence" whose other
    numbers ("guage 25 (red)") would otherwise be read as strain IDs.
    """
    field = parse_strain_field(details)
    if field:
        return field, []
    sentences = [s for s in _SENTENCE_SPLIT_PATTERN.split(details) if _STRAIN_WORD_PATTERN.search(s)]
    return field, sentences


def parse_yst_strain_ids(log_text: str) -> list[str]:
    """Extract numeric IDs from YST_XXXX references in the log text.

    Strains are written as e.g. "YST_1490" in experiment details or comments.
    Returns just the number part ("1490") so it can be looked up in the
    strain_tf_database.
    """
    return dedupe_preserve_order(_YST_STRAIN_PATTERN.findall(log_text))


def parse_strain_ids_from_details(details: str) -> list[str]:
    """Return numeric lab strain IDs written in the Experiment details block.

    Two sources, each with its own level of trust:

    1. The ``Strain:`` field is a strain list by definition, so every bare number
       in it counts — "Strain: 78 (BY4742), 1579 (BY4742 Morgan)" gives 78, 1579.
    2. Prose sentences mentioning strains, where only a number carrying a
       parenthetical description is trusted — "Strains are 87 (msn2-GFP) and
       1138 (msn2-mCherry)" gives 87, 1138.

    ``YST``-prefixed references are picked up anywhere in the block, since the
    prefix already disambiguates them.
    """
    if not details:
        return []
    strain_ids = _YST_STRAIN_PATTERN.findall(details)
    field, sentences = strain_description(details)

    # Source 1: the Strain: field, with residue ranges removed first
    if field:
        strain_ids.extend(
            _BARE_STRAIN_ID_PATTERN.findall(_GENE_PARENTHETICAL_PATTERN.sub(" ", field))
        )

    # Source 2: prose strain sentences, annotated numbers only
    for sentence in sentences:
        strain_ids.extend(_ANNOTATED_STRAIN_ID_PATTERN.findall(sentence))

    return dedupe_preserve_order(strain_ids)


def parse_strain_labels_from_details(details: str) -> list[str]:
    """Return the human-written strain labels listed in the ``Strain:`` field.

    One label per comma- or newline-separated item, e.g. "Hog1-GFP", "YST365 (FY4)".
    These carry no ID that joins to the strain database but are the only record of
    the strain for experiments that were never given a number.
    """
    field = parse_strain_field(details)
    if not field:
        return []
    return dedupe_preserve_order(re.split(r'[,\n]', field))


# Newer Batgirl logs list positions under a per-group heading instead of a
# Points: table:  "group: pH7_24 field: position" / "Name,X,Y,Z,..." / "pH7_24_001,..."
_POSITION_BLOCK_PATTERN = re.compile(
    r'(?im)^group\s*:\s*\S+\s+field\s*:\s*position\s*$\n^Name\s*,.*$\n((?:^[^\s,][^\n]*\n?)+)',
)

# Position names carry a trailing index: "YST_247_001" -> "YST_247", "by4741_01" -> "by4741"
_POSITION_INDEX_SUFFIX = re.compile(r'_\d+$')


def parse_position_names(text: str) -> list[str]:
    """Return imaged position names with their trailing index stripped.

    Positions are named after the strain they hold, in both the old multiDGUI
    ``Points:`` table and the newer per-group position blocks, so their prefixes
    are a strain record for experiments whose ``Strain:`` field just says
    "see position names" (datasets 593, 824, 1655).
    """
    names = []
    for block in _POSITION_BLOCK_PATTERN.findall(text):
        for line in block.splitlines():
            names.append(line.split(",")[0].strip())
    m = re.search(r'(?ims)^Points:\s*\n(.*?)(?:\n\s*\n|^Flow_control:|^Dynamic flow details:)', text)
    if m:
        rows = list(csv.reader(m.group(1).splitlines(), skipinitialspace=True))
        if rows and "position name" in [col.strip().lower() for col in rows[0]]:
            idx = [col.strip().lower() for col in rows[0]].index("position name")
            names.extend(row[idx].strip() for row in rows[1:] if len(row) > idx)
    return dedupe_preserve_order(_POSITION_INDEX_SUFFIX.sub("", n) for n in names if n)


def parse_strain_ids_from_positions(text: str) -> list[str]:
    """Return YST strain numbers taken from position names, e.g. ``YST_247_001`` → 247.

    Only ``YST``-prefixed names are read. A bare number in a position name is
    usually part of the condition rather than a strain — ``pH7_24_001`` is pH 7.24,
    not strain 7 or 24 — so nothing else here is trustworthy.
    """
    return dedupe_preserve_order(
        strain_id
        for name in parse_position_names(text)
        for strain_id in _YST_STRAIN_PATTERN.findall(name)
    )


def parse_point_groups(text: str) -> list[str]:
    """Return old acquisition point-group labels such as `group 1: by4741`.

    The multiDGUI `Points:` table lists one row per imaged position; positions
    belonging to the same group share a strain, so the position-name prefix
    (minus its trailing `_NN` index) names the strain.
    """
    m = re.search(r'(?ims)^Points:\s*\n(.*?)(?:\n\s*\n|^Flow_control:|^Dynamic flow details:)', text)
    if not m:
        return []
    rows = list(csv.reader(m.group(1).splitlines(), skipinitialspace=True))
    if not rows:
        return []
    header = [col.strip().lower() for col in rows[0]]
    if "position name" not in header or "group" not in header:
        return []
    name_idx = header.index("position name")
    group_idx = header.index("group")

    groups: dict[str, list[str]] = {}
    for row in rows[1:]:
        if len(row) <= max(name_idx, group_idx):
            continue
        position_name = row[name_idx].strip()
        group = row[group_idx].strip()
        prefix = re.sub(r'_\d+$', '', position_name).strip()  # "by4741_01" -> "by4741"
        if not group or not prefix:
            continue
        groups.setdefault(group, [])
        if prefix not in groups[group]:
            groups[group].append(prefix)
    return [f"group {group}: {'/'.join(names)}" for group, names in groups.items()]


def parse_strain_ids(log_text: str) -> Parsed[list[str]]:
    """Return strain/group IDs found in log or acquisition metadata.

    Numeric IDs come first because they are the only values that join to
    ``strain_tf_database.csv``; the human-written labels follow so that
    experiments without a numbered strain still record something.

    Order: numeric group IDs (``group: 898``), YST references, numeric IDs written
    in the Experiment details, YST numbers in position names, the ``Strain:``
    field labels, then old-format acquisition point groups.

    Every contributing extractor is named in the returned ``sources``.
    """
    details = parse_experiment_details(log_text)
    contributions = [
        ("group-lines", _STRAIN_PATTERN.findall(log_text)),
        ("yst-refs", parse_yst_strain_ids(log_text)),
        ("details", parse_strain_ids_from_details(details)),
        ("position-names", parse_strain_ids_from_positions(log_text)),
        ("strain-field-labels", parse_strain_labels_from_details(details)),
        ("point-groups", parse_point_groups(log_text)),
    ]
    values = [value for _, found in contributions for value in found]
    sources = tuple(name for name, found in contributions if found)
    return Parsed(dedupe_preserve_order(values), sources)


# ---------------------------------------------------------------------------
# Experimental condition
# ---------------------------------------------------------------------------

# Finds the "switch from X to Y" phrase inside the experiment details.
# Group 1 = the starting condition (X), group 2 = the ending condition (Y).
# The lookahead stops Y before punctuation, volume units (ul/ml/g), or
# context words like "YST_", "added" that begin unrelated follow-on clauses.
# Both sides are length-capped: a media description is short, and without the cap
# an unpunctuated details block let Y run to the end of the text
# ("...to low nitrogen mediaSorbitol replaces ammonium sulphate in the low N2 media").
_DETAILS_FIELD_LABEL = r'\b(?:Strain|Comments?|Aim|Notes?)\s*:'

# One character of a media description: anything that does not start the next
# field. Both sides of the phrase are built from it, because either can run away.
# Dataset 798's Aim ends mid-sentence — "Aim: switch from .05% Strain: 247
# Comments: switch from .05% glucose to 1%" — and an unguarded FROM side crossed
# two field labels to reach the "to" in the Comments.
_MEDIA_CHAR = r'(?:(?!\s*' + _DETAILS_FIELD_LABEL + r').)'

_SWITCH_CONDITION_PATTERN = re.compile(
    r'switch(?:ing)?\s+from\s+(' + _MEDIA_CHAR + r'{1,60}?)'
    r'\s+to\s+(' + _MEDIA_CHAR + r'{1,60}?)'
    r'(?=[.,;]\s|[.,;]$|\s+\d+\s*(?:ul|ml|g\b)|\s+YST_|\s+added\b'
    r'|\s*' + _DETAILS_FIELD_LABEL + r'|\Z)',
    re.IGNORECASE,
)

# Experimenters state a no-switch experiment plainly. Recognising it avoids
# paying for an LLM call only to be told there is no condition.
_NO_SWITCH_PATTERN = re.compile(
    r'\bno\s+switch(?:e[sd])?\b|\bno\s+media\s+change\b|\bwithout\s+switching\b',
    re.IGNORECASE,
)

_CONDITION_SYSTEM_PROMPT = """\
You are extracting the experimental condition from a microscopy experiment log.
The condition describes what changed during the experiment (e.g. a media switch).

Return the condition as a concise "X to Y" phrase on a single line, for example:
  2% glucose to 0% glucose
  high nitrogen to low nitrogen
  2% glucose to 0.1% glucose

If no condition change is described, return exactly: UNKNOWN
Return only the condition phrase or UNKNOWN — no other text.
"""


def parse_condition(
    log_text: str,
    client: "openai.OpenAI | None" = None,
    model: str = config.MODEL,
) -> Parsed[str]:
    """Extract the experimental condition from the 'Experiment details' section.

    Strategy:
    1. Locate the "Experiment details:" line and extract its content.
    2. Search for a "switch from X to Y" phrase and return "X to Y" directly.
       Example: "Switch from 2% glucose to 0%." → "2% glucose to 0%"
    3. If the details state outright that nothing was switched, stop.
    4. Otherwise ask the LLM to interpret the free-text details.

    An empty string means "no condition change", which is the correct answer for
    most experiments on this server: only 470 of 1710 datasets with a log mention
    a switch at all. Earlier versions fell back to returning the first sentence of
    the details, which filled 55% of the column with the experiment's *aim*
    ("Aim: Measure growth rate in raffinose") dressed up as its condition.
    """
    # Step 1: extract the experiment details block
    details = parse_experiment_details(log_text)
    if not details:
        return Parsed("", ())

    # Step 2: fast regex path — "switch from X to Y" is unambiguous
    sw = _SWITCH_CONDITION_PATTERN.search(details)
    if sw:
        from_part = sw.group(1).strip().rstrip(".,; ")  # e.g. "2% glucose"
        to_part = sw.group(2).strip().rstrip(".,; ")    # e.g. "0%"
        return Parsed(f"{from_part} to {to_part}", ("switch-phrase",))

    # Step 3: the details say there was no switch — believe them, and skip the LLM
    if _NO_SWITCH_PATTERN.search(details):
        return Parsed("", ("no-switch-stated",))

    # Step 4: LLM fallback — regex found no switch phrase; ask the LLM to
    # interpret the free-text experiment details. Only the details are sent,
    # not the whole log.
    llm_condition = llm.complete_or_none(
        client,
        _CONDITION_SYSTEM_PROMPT,
        details[:500],
        max_tokens=40,  # condition phrase is short
        model=model,
        label="condition",
    )
    if llm_condition and llm_condition.upper() != "UNKNOWN":
        return Parsed(llm_condition, ("llm",))

    return Parsed("", ())
