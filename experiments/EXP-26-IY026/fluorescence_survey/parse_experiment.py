"""Experiment-level fields parsed from log text: timepoints, strains, condition."""

import csv
import re

import openai

from . import config, llm
from .utils import dedupe_preserve_order

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
# Strain / group IDs
# ---------------------------------------------------------------------------

# Batgirl-format position lines: "group: 898 field: ..."
_STRAIN_PATTERN = re.compile(r'^group\s*:\s*(\d+)', re.IGNORECASE | re.MULTILINE)

# Free-text "Strain:" field inside the Experiment details block
_STRAIN_DETAILS_PATTERN = re.compile(
    r'Experiment details:.*?\bStrain:\s*(.*?)(?:\s+Comments:|\n)',
    re.IGNORECASE | re.DOTALL,
)

# Strains are sometimes referenced in log text as "YST_1490" (lab naming convention).
# \b ensures we don't match mid-word; (\d+) captures just the numeric part.
_YST_STRAIN_PATTERN = re.compile(r'\bYST_(\d+)\b', re.IGNORECASE)


def parse_yst_strain_ids(log_text: str) -> list[str]:
    """Extract numeric IDs from YST_XXXX references in the log text.

    Strains are written as e.g. "YST_1490" in experiment details or comments.
    Returns just the number part ("1490") so it can be looked up in the
    strain_tf_database.
    """
    return dedupe_preserve_order(_YST_STRAIN_PATTERN.findall(log_text))


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


def parse_strain_ids(log_text: str) -> list[str]:
    """Return strain/group IDs found in log or acquisition metadata.

    Captures numeric group IDs (e.g. ``group: 898``), YST_XXXX numeric parts,
    explicit ``Strain:`` fields, and old-format acquisition point groups.
    """
    values = _STRAIN_PATTERN.findall(log_text)
    values.extend(parse_yst_strain_ids(log_text))
    m = _STRAIN_DETAILS_PATTERN.search(log_text)
    if m:
        values.extend(part.strip() for part in m.group(1).split(","))
    values.extend(parse_point_groups(log_text))
    return dedupe_preserve_order(values)


# ---------------------------------------------------------------------------
# Experimental condition
# ---------------------------------------------------------------------------

# The "Experiment details:" line runs until the next section separator ("---...---")
# or the start of "Acquisition settings". re.DOTALL lets . match newlines so
# multi-line details are captured in one match.
_EXP_DETAILS_PATTERN = re.compile(
    r'Experiment details:\s*(.+?)(?=\n\s*-{3,}|\n\s*Acquisition settings|\Z)',
    re.IGNORECASE | re.DOTALL,
)

# Finds the "switch from X to Y" phrase inside the experiment details.
# Group 1 = the starting condition (X), group 2 = the ending condition (Y).
# The lookahead stops Y before punctuation, volume units (ul/ml/g), or
# context words like "YST_", "added" that begin unrelated follow-on clauses.
_SWITCH_CONDITION_PATTERN = re.compile(
    r'switch(?:ing)?\s+from\s+(.+?)\s+to\s+(.+?)'
    r'(?=[.,;]\s|[.,;]$|\s+\d+\s*(?:ul|ml|g\b)|\s+YST_|\s+added\b|\Z)',
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
) -> str:
    """Extract the experimental condition from the 'Experiment details' section.

    Strategy:
    1. Locate the "Experiment details:" line and extract its content.
    2. Search for a "switch from X to Y" phrase and return "X to Y" directly.
       Example: "Switch from 2% glucose to 0%." → "2% glucose to 0%"
    3. If no switch phrase is found and an LLM client is provided, ask the LLM
       to interpret the free-text details and extract the condition.
    4. Last resort: return the first sentence of the details (capped at 200 chars).
    """
    # Step 1: extract the experiment details block
    m = _EXP_DETAILS_PATTERN.search(log_text)
    if not m:
        return ""
    details = m.group(1).strip()
    if not details:
        return ""

    # Step 2: fast regex path — "switch from X to Y" is unambiguous
    sw = _SWITCH_CONDITION_PATTERN.search(details)
    if sw:
        from_part = sw.group(1).strip().rstrip(".,; ")  # e.g. "2% glucose"
        to_part = sw.group(2).strip().rstrip(".,; ")    # e.g. "0%"
        return f"{from_part} to {to_part}"

    # Step 3: LLM fallback — regex found no switch phrase; ask the LLM to
    # interpret the free-text experiment details. Only the details are sent,
    # not the whole log.
    llm_condition = llm.complete_or_none(
        client,
        _CONDITION_SYSTEM_PROMPT,
        details[:500],
        max_tokens=40,  # condition phrase is short
        model=model,
    )
    if llm_condition and llm_condition.upper() != "UNKNOWN":
        return llm_condition

    # Step 4: last resort — no switch phrase and no LLM; return first sentence
    first_sentence = re.split(r'(?<=[.!?])\s', details)[0]
    return first_sentence[:200]
