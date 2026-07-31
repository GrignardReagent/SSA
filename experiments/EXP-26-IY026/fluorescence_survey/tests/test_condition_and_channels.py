"""Condition extraction and channel parsing.

Log excerpts are taken verbatim from datasets on the lab OMERO server.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fluorescence_survey.parse_channels import (  # noqa: E402
    fluorescence_channels,
    parse_channels,
    parse_image_config_channels,
    parse_runtime_channels,
)
from fluorescence_survey.parse_experiment import (  # noqa: E402
    parse_condition,
    parse_experiment_details,
    parse_strain_ids_from_positions,
)


def log(details: str) -> str:
    return (
        "Swain Lab microscope experiment log file\n"
        "-------------------------------------\n"
        f"Experiment details: {details}\n"
        "-------------------------------------\n"
        "Acquisition settings\n"
    )


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

def test_switch_phrase_is_extracted():
    parsed = parse_condition(log("Aim: stress  Comments: Switch from 2% glucose to 0% glucose."))
    assert parsed.value == "2% glucose to 0% glucose"
    assert parsed.label() == "switch-phrase"


def test_unpunctuated_switch_is_left_to_the_model():
    """Without the length cap the 'to' side ran to the end of the details block
    ('...to low nitrogen mediaSorbitol replaces ammonium sulphate...'). Capped, the
    regex declines and — in a real run, with a client — step 4 handles it."""
    parsed = parse_condition(log(
        "Switch from high nitrogen media to low nitrogen mediaSorbitol replaces "
        "ammonium sulphate in the low N2 media and this is a long trailing clause"
    ), client=None)
    assert parsed.value == ""
    assert parsed.label() == "none"


def test_switch_phrase_stops_at_the_next_field_label():
    """Dataset 798 runs two fields together: the phrase must not swallow 'Strain: 247'."""
    parsed = parse_condition(log(
        "Aim: growth  switch from 2% to .05% Strain: 247 Comments: switch from .05% glucose to 1%"
    ))
    assert parsed.value == "2% to .05%"


def test_no_switch_stated_gives_no_condition():
    """Dataset 692 says so outright; there is nothing for the LLM to add."""
    parsed = parse_condition(log(
        "Aim: Growth rate/bud timing measurement using BABY Strain: YST_625 "
        "Comments: No switch. Media is SC+2% pyruvate 4ul/min"
    ))
    assert parsed.value == ""
    assert parsed.label() == "no-switch-stated"


def test_experiment_without_a_condition_returns_empty_not_its_aim():
    """Dataset 561: the aim is not the condition, and no LLM client is available."""
    parsed = parse_condition(log(
        "Aim: Measure growth rate in raffinose using BABY  Strain:   Comments:"
    ), client=None)
    assert parsed.value == ""
    assert parsed.label() == "none"


def test_details_stop_before_omero_bookkeeping():
    """Dataset 832 has no 'Microscope setup' heading and ran straight into OMERO fields."""
    details = parse_experiment_details(log(
        "Aim: Control for wild type growth  Strain: 77  Comments:\n"
        "Omero project:\nShampoo\nOmero tags:\n13-Sep-2017,Batgirl,"
    ))
    assert "Shampoo" not in details
    assert details.startswith("Aim: Control for wild type growth")


def test_hand_typed_duplicate_label_is_stripped():
    """Datasets 2806/2808/2810 begin 'Experiment details: Experiment details: ...'."""
    details = parse_experiment_details(log("Experiment details: localization of GAL2 in sugars"))
    assert details == "localization of GAL2 in sugars"


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def test_runtime_prose_is_not_a_channel():
    """'Channel does not use Smart EM camera mode.' put a channel named 'does' in 127
    of 220 sampled datasets."""
    text = (
        "Channel does not use Smart EM camera mode. Settings remain unchanged.\n"
        "Channel: GFP\n"
        "Channel configuration set to: mCherry\n"
    )
    assert parse_runtime_channels(text) == ["GFP", "mCherry"]


def test_image_config_block_stops_at_the_next_table():
    """Dataset 1687: 'Device properties:' and the position list follow with no blank line."""
    text = (
        "Image config,Channel,Description,Exposure (ms),Number of Z sections,Z spacing (um),Sectioning method\n"
        "brightfield1,Brightfield,Default bright field config,30,5,0.6,PIFOC\n"
        "pHluorin405_0_4,pHluorin405,Phluorin excitation from 405 LED,5,1,0.6,PIFOC\n"
        "Device properties:\n"
        "Image config,device,property,value\n"
        "pHluorin405_0_4,DTOL-DAC-1,Volts,0.4\n"
    )
    assert parse_image_config_channels(text) == [
        "brightfield1", "Brightfield", "pHluorin405_0_4", "pHluorin405",
    ]


HEADER = (
    "Image config,Channel,Description,Exposure (ms),"
    "Number of Z sections,Z spacing (um),Sectioning method\n"
)


def test_position_names_never_reach_the_channel_list():
    text = (
        HEADER
        + "brightfield1,Brightfield,Default bright field config,30,5,0.6,PIFOC\n"
        "group: pH7_24 field: position\n"
        "Name,X,Y,Z,Autofocus offset\n"
        "pH7_24_001,-5625,-3698,2263.35,116\n"
    )
    assert parse_channels(text) == ["brightfield1", "Brightfield"]


def test_device_properties_table_still_names_channels():
    """Dataset 615: GFP is named nowhere else in the log but this table's first column."""
    text = (
        HEADER
        + "brightfield1,Brightfield,Default bright field config,30,5,0.6,PIFOC\n"
        "Device properties:\n"
        "Image config,device,property,value\n"
        "GFP,DTOL-DAC-2,Volts,4\n"
        "2023-02-02 16:53:47,971 - INFO \n"
        "group: ch1 field: position\n"
    )
    assert parse_channels(text) == ["brightfield1", "Brightfield", "GFP"]


def test_hardware_device_names_are_not_channels():
    """In the device table column 1 is a DAC, not a channel."""
    text = (
        "Image config,device,property,value\n"
        "pHluorin405_0_4,DTOL-DAC-1,Volts,0.4\n"
        "pHluorin488_0_4,DTOL-DAC-2,Volts,0.4\n"
    )
    assert parse_channels(text) == ["pHluorin405_0_4", "pHluorin488_0_4"]


def test_channel_rows_interleaved_with_sub_blocks():
    """Dataset 1088 puts a 'Z settings:' block between every channel row."""
    text = (
        HEADER
        + "brightfield1,Brightfield,Default bright field config,30,\n"
        "Z settings: \n"
        "sections,spacing,method\n"
        "5,0.6,PIFOC\n"
        "GFP,GFPFast,Default GFP,30,\n"
        "Device: DTOL-DAC-2 Property: Volts Value: 4  \n"
        "Z settings: \n"
        "sections,spacing,method\n"
        "5,0.6,PIFOC\n"
        "mCherry,mCherry,mCherry imaging,100,\n"
        "group: YST_605 field: position\n"
    )
    assert parse_channels(text) == [
        "brightfield1", "Brightfield", "GFP", "GFPFast", "mCherry",
    ]


def test_yfp_counts_as_fluorescence():
    """Dataset 932 acquires DIC + YFP over 200 timepoints."""
    assert fluorescence_channels(["DIC", "YFP"]) == ["YFP"]


def test_transmitted_light_is_not_fluorescence():
    assert fluorescence_channels(["Brightfield", "brightfield1", "DIC"]) == []


# ---------------------------------------------------------------------------
# Strain IDs from position names
# ---------------------------------------------------------------------------

def test_yst_position_names_give_strain_ids():
    """Dataset 1685: positions are named after the strain, with a trailing index."""
    text = (
        "group: YST_247 field: position\n"
        "Name,X,Y,Z,Autofocus offset\n"
        "YST_247_001,-8968,-3698,2263.35,116\n"
        "YST_247_002,-8953,-3698,2262.22,116\n"
    )
    assert parse_strain_ids_from_positions(text) == ["247"]


def test_condition_encoded_in_position_names_is_not_a_strain_id():
    """'pH7_24_001' is pH 7.24, not strain 7 or strain 24."""
    text = (
        "group: pH7_24 field: position\n"
        "Name,X,Y,Z,Autofocus offset\n"
        "pH7_24_001,-5625,-3698,2263.35,116\n"
    )
    assert parse_strain_ids_from_positions(text) == []


def test_switch_phrase_does_not_cross_a_field_label_backwards():
    """Dataset 798's Aim ends mid-sentence, so the FROM side must not run on
    through 'Strain:' and 'Comments:' to reach the 'to' in the next field."""
    parsed = parse_condition(log(
        "Aim: switch from .05% Strain: 247   Comments: switch from .05% glucose to 1%. "
        "Not osmotically balanced."
    ))
    assert parsed.value == ".05% glucose to 1%"


def test_truncated_final_line_is_dropped():
    """The 250 kB cap lands mid-token; 'Channel: Brightfield' cut short registered
    a channel called 'Brigh' on nine datasets."""
    from fluorescence_survey.omero_source import _read_file_annotation_text

    class FakeAnnotation:
        def getFileInChunks(self):
            yield b"Channel: GFP\nChannel: mCherry\nChannel: Brightfield\n"

    text, truncated = _read_file_annotation_text(FakeAnnotation(), max_bytes=35)
    assert truncated
    assert text == "Channel: GFP\nChannel: mCherry\n"
    assert parse_runtime_channels(text) == ["GFP", "mCherry"]


def test_coumarin_counts_as_fluorescence():
    """Datasets 1239 and 2711 acquire Brightfield + coumarin over 240 timepoints."""
    assert fluorescence_channels(["Brightfield", "coumarin"]) == ["coumarin"]


def test_upload_records_are_not_metadata():
    """Both wordings appear: dataset 677 keeps an 'upload completed' record alongside
    the 'upload failed' one, and neither is experimental metadata."""
    from fluorescence_survey.omero_source import _is_upload_housekeeping

    assert _is_upload_housekeeping("Upload to staffa is in progress")
    assert _is_upload_housekeeping(
        "Swain lab dataset Omero upload completed.\r\nUpload by: SCE-BIO-C04521\r\n"
    )
    assert _is_upload_housekeeping("Swain lab dataset Omero upload failed, \r\nServer: islay")
    assert not _is_upload_housekeeping("Swain Lab microscope experiment log file\nExperiment details:")
