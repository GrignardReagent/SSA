"""Channel-name extraction from the four metadata layouts seen on the server.

`parse_channels` runs all four extractors and merges their results; the log
formats changed over the years, so a single dataset may match any one of them.
"""

import csv
import re

from .utils import dedupe_preserve_order
from .vocabulary import FLUORESCENCE_CHANNELS

# Runtime acquisition lines such as "Channel: GFP" or "Image config: brightfield"
_CHANNEL_LOG_PATTERN = re.compile(
    r'(?im)^\s*(?:Channel(?: configuration set to)?|Image config)\s*:?\s*([A-Za-z][\w-]*)'
)


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


def parse_image_config_channels(text: str) -> list[str]:
    """Return channel/config names from newer `Image Configs` CSV blocks."""
    channels = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.lower().startswith("image config,channel,"):
            continue
        # Consume data rows until the block ends at a blank line
        for data_line in lines[i + 1:]:
            if not data_line.strip():
                break
            row = next(csv.reader([data_line], skipinitialspace=True), [])
            if len(row) < 2:
                continue
            channels.extend([row[0].strip(), row[1].strip()])
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
    return dedupe_preserve_order(channels)


def fluorescence_channels(channels: list[str]) -> list[str]:
    """Return the subset of `channels` that are known fluorescence channels.

    Matching is exact first, then substring, so variants such as "GFP_1" or
    "BrightfieldGFP" still resolve to their canonical channel name.
    """
    fluorescence = []
    known = {channel.lower(): channel for channel in FLUORESCENCE_CHANNELS}
    for channel in channels:
        lower_channel = channel.lower()
        if lower_channel in ("brightfield", "brightfield1"):
            continue
        if lower_channel in known:
            fluorescence.append(known[lower_channel])
            continue
        for known_lower, known_name in known.items():
            if known_lower in lower_channel:
                fluorescence.append(known_name)
                break
    return dedupe_preserve_order(fluorescence)
