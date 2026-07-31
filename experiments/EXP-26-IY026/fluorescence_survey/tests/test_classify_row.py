"""End-to-end: a log file in, a results.csv row out.

The two datasets below are the ones this pipeline gets wrong in opposite ways, so
they are worth pinning: 615 records its strains only in prose, and 692 records
outright that nothing was switched.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fluorescence_survey.classification import RESULT_FIELDS, classify  # noqa: E402
from fluorescence_survey.omero_source import MetadataAnnotations  # noqa: E402

DATASET_615_LOG = """Swain Lab microscope experiment log file
-------------------------------------
Experiment details: Timelapse test with SID device. Strains are 87 (msn2-GFP) and 1138 \
(msn2-mCherry). Here switching from 2% glucose to 0% glucose.
Microscope setup for used channels:
GFP:
470nm LED
-------------------------------------
Acquisition settings
Image config,Channel,Description,Exposure (ms),Number of Z sections,Z spacing (um),Sectioning method
brightfield1,Brightfield,Default bright field config,30,5,0.6,PIFOC
Device properties:
Image config,device,property,value
GFP,DTOL-DAC-2,Volts,4
Number of timepoints = 18
"""

DATASET_692_LOG = """Swain Lab microscope experiment log file
-------------------------------------
Experiment details: Aim: Growth rate/bud timing measurement using BABY Strain: YST_625 \
Comments: No switch. Media is SC+2% pyruvate 4ul/min
Omero project:
BABY paper
-------------------------------------
Acquisition settings
Image config,Channel,Description,Exposure (ms),Number of Z sections,Z spacing (um),Sectioning method
brightfield1,Brightfield,Default bright field config,30,5,0.6,PIFOC
Number of timepoints = 180
group: ch1_MSN2 field: position
"""


def row_for(log_text: str, dataset_id: int, dataset_name: str) -> dict:
    """Run the full classifier with no model, so only the deterministic paths fire."""
    return classify(
        None, dataset_id, dataset_name,
        MetadataAnnotations(log_text=log_text, log_filename="test.log"),
    )


def test_row_has_exactly_the_declared_columns():
    assert set(row_for(DATASET_615_LOG, 615, "SID_test_01")) == set(RESULT_FIELDS)


def test_strains_and_tf_read_out_of_prose():
    row = row_for(DATASET_615_LOG, 615, "SID_test_01")
    assert row["strain_id"] == "87;1138"
    assert row["tf_identity"] == "Msn2"
    assert row["condition"] == "2% glucose to 0% glucose"


def test_channel_named_only_in_the_device_table_still_classifies():
    row = row_for(DATASET_615_LOG, 615, "SID_test_01")
    assert row["channels"] == "GFP"
    assert row["all_channels"] == "brightfield1;Brightfield;GFP"
    assert row["classification"] == "YES"


def test_no_switch_gives_an_empty_condition_not_the_aim():
    row = row_for(DATASET_692_LOG, 692, "BABY_pyruvate_02")
    assert row["condition"] == ""
    assert "condition=no-switch-stated" in row["provenance"]


def test_brightfield_only_is_not_a_fluorescence_timelapse():
    row = row_for(DATASET_692_LOG, 692, "BABY_pyruvate_02")
    assert row["all_channels"] == "brightfield1;Brightfield"
    assert row["classification"] == "NO"


def test_provenance_names_the_source_of_every_field():
    row = row_for(DATASET_692_LOG, 692, "BABY_pyruvate_02")
    assert row["provenance"] == (
        "condition=no-switch-stated | "
        "strain=yst-refs,details,strain-field-labels | "
        "tf=group-labels | "
        "fluorescence=no-model | "
        "tf_localisation=known-tf"
    )


def test_no_model_configured_does_not_raise():
    """A brightfield-only dataset would otherwise reach the fluorescence LLM call."""
    assert row_for(DATASET_692_LOG, 692, "BABY_pyruvate_02")["raw_llm_response"] == ""
