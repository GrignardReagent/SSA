"""Strain ID and TF extraction from the free-text 'Experiment details' block.

Every log excerpt below is taken verbatim from a dataset on the lab OMERO server;
the dataset ID is given so the source can be checked.
"""

import sys
from pathlib import Path

# The package lives one level above this tests/ directory
sys.path.append(str(Path(__file__).resolve().parents[2]))

from fluorescence_survey.parse_experiment import (  # noqa: E402
    parse_experiment_details,
    parse_strain_field,
    parse_strain_ids,
    parse_strain_ids_from_details,
    parse_yst_strain_ids,
)
from fluorescence_survey.parse_tf import parse_tf_from_details  # noqa: E402


def log(details: str) -> str:
    """Wrap a details block in the surrounding Swain-lab log structure."""
    return (
        "Swain Lab microscope experiment log file\n"
        "-------------------------------------\n"
        f"Experiment details: {details}\n"
        "-------------------------------------\n"
        "Acquisition settings\n"
    )


# ---------------------------------------------------------------------------
# The details block itself
# ---------------------------------------------------------------------------

def test_details_block_stops_before_microscope_configuration():
    """The machine-written config dump must not reach the free-text parsers."""
    details = parse_experiment_details(log(
        "Aim: Grow cells  Strain: 87  Comments:\n"
        "Microscope setup for used channels:\n"
        "GFP:\nMicromanager config file:C:\\Batgirl_11_10_22_flavin.txt\n"
    ))
    assert details == "Aim: Grow cells  Strain: 87  Comments:"


def test_details_block_absent_returns_empty_string():
    assert parse_experiment_details("no experiment details here") == ""


# ---------------------------------------------------------------------------
# Strain IDs from prose (dataset 615 / 697: no Strain: field at all)
# ---------------------------------------------------------------------------

DATASET_615_DETAILS = (
    "Timelapse test with SID device. Strains are 87 (msn2-GFP) and 1138 "
    "(msn2-mCherry). Pre-calibratedthe flow rates - recorded in a text file - "
    "using happy media with cy5 (2% glc in SC). Here switching to 0% glucosein "
    "SC in pairs of chambers sequentially."
)


def test_prose_strain_sentence_yields_both_ids():
    assert parse_strain_ids_from_details(DATASET_615_DETAILS) == ["87", "1138"]


def test_prose_strain_sentence_yields_tf():
    assert parse_tf_from_details(DATASET_615_DETAILS) == ["Msn2"]


def test_full_strain_id_chain_on_dataset_615():
    """End to end: the log text in, the results.csv value and its provenance out."""
    parsed = parse_strain_ids(log(DATASET_615_DETAILS))
    assert ";".join(parsed.value) == "87;1138"
    assert parsed.label() == "details"


def test_numbers_outside_the_strain_sentence_are_ignored():
    """'2% glc', 'cy5' and the flow rates must not become strain IDs."""
    ids = parse_strain_ids_from_details(DATASET_615_DETAILS)
    assert set(ids) == {"87", "1138"}


def test_prose_number_without_a_description_is_not_a_strain_id():
    """Outside the Strain: field, only '87 (msn2-GFP)' shaped mentions count."""
    assert parse_strain_ids_from_details("Strains were grown for 24 hours.") == []


# ---------------------------------------------------------------------------
# Strain IDs from the structured Strain: field
# ---------------------------------------------------------------------------

def test_bare_numbered_strain_list():
    """Dataset 682."""
    details = "Aim: Compare stocks  Strain: 78 (BY4742), 1579 (BY4742 Morgan), 1580 (tsa1 tsa2 del)  Comments:"
    assert parse_strain_ids_from_details(details) == ["78", "1579", "1580"]


def test_yst_prefix_variants_are_reduced_to_the_number():
    """Datasets 489, 490, 692, 1340, 1621 — YST_1490, YST365, YST-708, yst556."""
    assert parse_yst_strain_ids("YST_1490 YST365 YST-708 yst556") == ["1490", "365", "708", "556"]


def test_background_strain_names_are_not_strain_ids():
    """Dataset 1239: BY4742 and W303 name the background, 78 and 56 the strains."""
    details = "Strain: By4742 (78) W303 Ade2+ (56) Comments:"
    assert parse_strain_ids_from_details(details) == ["78", "56"]


def test_position_ranges_are_not_strain_ids():
    """Dataset 1621: 'pos001-006' must not contribute 006."""
    details = "Strain: pos001-006 yst365 prototroph, pos007-012 yst556 whi5-mCherry Comments:"
    assert parse_strain_ids_from_details(details) == ["365", "556"]


def test_residue_ranges_are_not_strain_ids():
    """Dataset 1261: Msn2(604-636) is a truncation, not strain 604."""
    details = "Strain: Msn2(604-636)-WTI-GFP\nDot6(282)-dIp-dEp-GFP\nComments:"
    assert parse_strain_ids_from_details(details) == []


def test_strain_field_ends_at_the_next_label_even_without_a_space():
    """Dataset 2554: 'Strain:247Comments: Omero tags: 13-Sep-2017,...'."""
    details = "Strain:247Comments: Omero tags: 13-Sep-2017,Batgirl,pHluorin,247,pH"
    assert parse_strain_field(details) == "247"
    assert parse_strain_ids_from_details(details) == ["247"]


def test_strain_field_wins_over_prose_in_the_same_sentence():
    """Dataset 907: 'Strain: none' plus 'guage 25 (red)' must yield nothing."""
    details = (
        "Aim: Protocol for switching with T-junction Strain: none "
        "Comments: Metal tube removed from a blunt syringe tip, guage 25 (red)."
    )
    assert parse_strain_ids_from_details(details) == []


# ---------------------------------------------------------------------------
# TF identity from the Strain: field
# ---------------------------------------------------------------------------

def test_untagged_tf_names_in_the_strain_field():
    """Dataset 1575: no fluorescent tag to key off, just the TF names."""
    details = "Strain: Mig1, Dot6, Maf1, Sfp1, Msn2 Comments:"
    assert parse_tf_from_details(details) == ["Mig1", "Dot6", "Maf1", "Sfp1", "Msn2"]


def test_numbered_strains_with_mixed_tagged_and_bare_tf_names():
    """Dataset 826: only Msn2 carries a tag; the rest are bare."""
    details = "Strain: 87 (Msn2-GFP), 416 (Hog1), 424 (Dot6), 429(Yap1), 430 (Sfp1) Comments:"
    assert parse_strain_ids_from_details(details) == ["87", "416", "424", "429", "430"]
    assert parse_tf_from_details(details) == ["Msn2", "Hog1", "Dot6", "Yap1", "Sfp1"]


def test_dual_reporter_strains_separated_by_slashes():
    """Dataset 1552."""
    details = "Strain: Msn2-GFP/Dot6-mCherry\nSfp1-GFP/Maf1-mCherry\nComments:"
    assert parse_tf_from_details(details) == ["Msn2", "Dot6", "Sfp1", "Maf1"]


def test_non_tf_markers_do_not_become_tfs():
    """Dataset 673: Ura7/Ura8 aggregation reporters are not TFs."""
    details = "Strain: Ura7H360A-GFP, Ura8-GFP, Ura8H360R-GFP Comments:"
    assert parse_tf_from_details(details) == []


def test_plasmid_and_background_names_do_not_become_tfs():
    """Dataset 1244: pZTIR, AID and colony labels are not gene names."""
    details = "Strain: 1312 (Mig1-mCherry, control, with GFP-AID), 1284 (Snf1-AID) Comments:"
    assert parse_tf_from_details(details) == ["Mig1", "Snf1"]


def test_details_without_a_strain_description_yield_nothing():
    assert parse_tf_from_details("Aim: Image of graticule slide for calibration") == []
    assert parse_strain_ids_from_details("") == []
