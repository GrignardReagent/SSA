"""Channel-name extraction from the four metadata layouts seen on the server.

`parse_channels` runs all four extractors and merges their results; the log
formats changed over the years, so a single dataset may match any one of them.
"""

import csv
import re

from .utils import dedupe_preserve_order
from .vocabulary import BRIGHTFIELD_CHANNELS, FLUORESCENCE_CHANNELS

# Runtime acquisition lines such as "Channel: GFP" or "Channel configuration set
# to: GFP". The colon is mandatory: without it the pattern also matched ordinary
# prose in the log ("Channel does not use Smart EM camera mode."), which put a
# channel named "does" into 127 of the 220 datasets sampled.
_CHANNEL_LOG_PATTERN = re.compile(
    r'(?im)^\s*(?:Channel(?: configuration set to)?|Image config)\s*:\s*([A-Za-z][\w-]*)'
)

# A channel name is a single bare word: no spaces, no punctuation beyond _ and -,
# and short. Applied as a last guard so that a table parser running off the end of
# its block cannot put stage coordinates, timestamps or position names into the
# channel list.
_CHANNEL_NAME_SHAPE = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,23}$')


def parse_acq_channels(text: str) -> list[str]:
    """Return channel names from an old multiDGUI `Channels:` table."""
    m = re.search(
        r'(?ims)^(?:.*Acq\.txt\s+)?Channels:\s*\n(.*?)'
        r'(?:\n\s*\n|^Z_sectioning:|^Time_settings:|^Points:)',
        text,
    )
    if not m:
        return []
    channels = []
    # The block is a CSV table whose first column holds the channel name
    for row in csv.reader(m.group(1).splitlines(), skipinitialspace=True):
        if not row:
            continue
        name = row[0].strip()
        if not name or name.lower().startswith("channel name"):
            continue  # skip the header row
        channels.append(name)
    return dedupe_preserve_order(channels)


# The acquisition-settings region ends where the per-timepoint log begins: either a
# "group: ..." block or a timestamped INFO line.
_ACQUISITION_REGION_END = re.compile(r'^\s*(?:group\s*:|\d{4}-\d{2}-\d{2}[ T]\d{2}:)')


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def parse_image_config_channels(text: str) -> list[str]:
    """Return channel/config names from newer `Image Configs` CSV blocks.

    Two different tables share the ``Image config,...`` header and the header says
    which columns hold channel names:

    - ``Image config,Channel,Description,Exposure (ms),...`` — columns 0 and 1 are
      the config and the channel it drives (``pHluorin405_0_4,pHluorin405``).
    - ``Image config,device,property,value`` — column 1 is a piece of hardware
      (``DTOL-DAC-2``), so only column 0 is a channel. On dataset 615 this is the
      only place ``GFP`` is named at all.

    Rows are filtered rather than the block being delimited, because the layout is
    not consistently delimited: dataset 1088 interleaves ``Z settings:`` sub-blocks
    between channel rows, while dataset 1687 runs the position list on with no
    blank line. A channel row in the first table is recognised by its numeric
    exposure column; the scan stops where the per-timepoint log starts.
    """
    channels = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        header = [col.strip().lower() for col in next(csv.reader([line], skipinitialspace=True), [])]
        if not header or header[0] != "image config":
            continue
        header_names_channel = len(header) > 1 and header[1] == "channel"
        for data_line in lines[i + 1:]:
            if _ACQUISITION_REGION_END.match(data_line):
                break
            row = [field.strip() for field in next(csv.reader([data_line], skipinitialspace=True), [])]
            if row and row[0].lower() == "image config":
                break  # the next table starts here; the outer loop handles it
            if len(row) < 2 or not _CHANNEL_NAME_SHAPE.match(row[0]):
                continue  # a sub-block heading or a numeric settings row
            if not header_names_channel:
                channels.append(row[0])
            elif len(row) >= 4 and _is_number(row[3]):  # the exposure column
                channels.extend([row[0], row[1]])
    return dedupe_preserve_order(channels)


def parse_setup_channels(text: str) -> list[str]:
    """Return channel section names from `Microscope setup for used channels` blocks."""
    m = re.search(
        r'(?ims)^Microscope setup for used channels:\s*\n(.*?)'
        r'(?:^Micromanager config file:|^Omero project:|^Experiment started at:)',
        text,
    )
    if not m:
        return []
    # Inside the block, each channel is its own "Name:" sub-heading on one line
    return dedupe_preserve_order(
        match.group(1).strip()
        for match in re.finditer(r'(?m)^\s*([A-Za-z][\w-]*)\s*:\s*$', m.group(1))
    )


def parse_runtime_channels(text: str) -> list[str]:
    """Return channels mentioned in runtime acquisition lines."""
    return dedupe_preserve_order(
        channel for channel in _CHANNEL_LOG_PATTERN.findall(text) if channel.lower() != "name"
    )


def parse_channels(text: str) -> list[str]:
    """Return all channel names found across old and new metadata formats."""
    channels = []
    channels.extend(parse_acq_channels(text))
    channels.extend(parse_image_config_channels(text))
    channels.extend(parse_setup_channels(text))
    channels.extend(parse_runtime_channels(text))
    return [c for c in dedupe_preserve_order(channels) if _CHANNEL_NAME_SHAPE.match(c)]


def fluorescence_channels(channels: list[str]) -> list[str]:
    """Return the subset of `channels` that are known fluorescence channels.

    Matching is exact first, then substring, so variants such as "GFP_1" or
    "BrightfieldGFP" still resolve to their canonical channel name.
    """
    fluorescence = []
    known = {channel.lower(): channel for channel in FLUORESCENCE_CHANNELS}
    brightfield = {channel.lower() for channel in BRIGHTFIELD_CHANNELS}
    for channel in channels:
        lower_channel = channel.lower()
        if lower_channel in brightfield:
            continue
        if lower_channel in known:
            fluorescence.append(known[lower_channel])
            continue
        for known_lower, known_name in known.items():
            if known_lower in lower_channel:
                fluorescence.append(known_name)
                break
    return dedupe_preserve_order(fluorescence)
