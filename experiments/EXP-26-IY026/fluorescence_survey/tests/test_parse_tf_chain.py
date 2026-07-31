"""The TF identity chain: what short-circuits, what merges, and where each answer came from.

Log excerpts are taken verbatim from datasets on the lab OMERO server.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fluorescence_survey.parse_tf import (  # noqa: E402
    parse_tf_from_group_labels,
    parse_tf_from_groups,
    parse_tf_from_position_names,
)


def log(details: str, tail: str = "") -> str:
    """Wrap a details block in the surrounding Swain-lab log structure."""
    return (
        "Swain Lab microscope experiment log file\n"
        "-------------------------------------\n"
        f"Experiment details: {details}\n"
        "Microscope setup for used channels:\n"
        "GFP:\n470nm LED\n"
        f"{tail}\n"
        "-------------------------------------\n"
        "Acquisition settings\n"
    )


# ---------------------------------------------------------------------------
# Group labels — the four shapes the lab writes them in
# ---------------------------------------------------------------------------

def test_chamber_prefixed_label():
    """`ch1_REI1`: whatever follows the chamber prefix is the TF, vocabulary or not."""
    assert parse_tf_from_group_labels("group: ch1_REI1 field: position") == ["Rei1"]


def test_tf_first_label():
    """Dataset 1006: `ACE2_ch17` puts the TF before the chamber."""
    text = "group: ACE2_ch17 field: position\ngroup: BAS1_ch6 field: position\n"
    assert parse_tf_from_group_labels(text) == ["Ace2", "Bas1"]


def test_mixed_case_chamber_labels():
    """The lab writes `ch10_Gcd1` as often as `ch1_ADE6`; an upper-case-only rule
    silently skipped most of them."""
    assert parse_tf_from_group_labels("group: ch10_Gcd1\ngroup: ch13_Asn2\n") == ["Gcd1", "Asn2"]


def test_chamber_labels_do_not_need_the_vocabulary():
    """Dataset 926 screens 19 chambers, six of them TFs absent from KNOWN_TFS."""
    text = "group: Aft1_ch18 field: position\ngroup: Cad1_ch17 field: position\n"
    assert parse_tf_from_group_labels(text) == ["Aft1", "Cad1"]


def test_plate_well_between_chamber_and_gene():
    """Datasets 1666-2462 label chambers `ch10_C11_RPS18A`. Reading the first token
    after the chamber reported the *well* (`C11`) as the TF."""
    text = (
        "group: ch10_C11_RPS18A field: position\n"
        "group: ch12_D1_GPM1 field: position\n"
        "group: ch17_d7_EDC1 field: position\n"
    )
    assert parse_tf_from_group_labels(text) == ["Rps18a", "Gpm1", "Edc1"]


def test_a_well_is_not_mistaken_for_a_short_gene_name():
    """`ch20_Msn2_GFP` must not read `Msn2` as a well and `GFP` as the gene."""
    assert parse_tf_from_group_labels("group: ch20_Msn2_GFP field: position") == ["Msn2"]


def test_position_names_are_read_as_chamber_labels():
    """Dataset 824: the Strain: field says only 'see position names'."""
    text = (
        "Points:\n"
        "Position name, X position, Y position, Group\n"
        "Dot6_001,-100,200,1\n"
        "Dot6_002,-101,200,1\n"
        "Hog1_001,-200,300,2\n"
        "\n"
    )
    assert parse_tf_from_position_names(text) == ["Dot6", "Hog1"]


def test_generic_position_names_name_no_tf():
    """Dataset 469 names its positions pos001..pos020."""
    text = (
        "Points:\n"
        "Position name, X position, Y position, Group\n"
        "pos001,-100,200,1\n"
        "pos002,-101,200,1\n"
        "\n"
    )
    assert parse_tf_from_position_names(text) == []


def test_tf_plus_tag_label():
    """Datasets 2605, 2612: `Msn2_GFP` names the construct, not the chamber."""
    text = "group: Msn2_GFP field: position\ngroup: Mig1_GFP_1 field: position\n"
    assert parse_tf_from_group_labels(text) == ["Msn2", "Mig1"]


def test_bare_name_label():
    assert parse_tf_from_group_labels("group: Dot6 field: position") == ["Dot6"]


def test_chamber_only_label_names_no_tf():
    """Dataset 615: `ch1`..`ch17` carry no strain information."""
    assert parse_tf_from_group_labels("group: ch1 field: position\ngroup: ch2 field: time") == []


def test_numeric_label_names_no_tf():
    """Dataset 2493: `group: 1352` is a strain ID, resolved via the strain DB instead."""
    assert parse_tf_from_group_labels("group: 1352 field: position") == []


def test_non_tf_markers_still_come_through_group_labels():
    """`ch1_VPH1` must yield Vph1 so the localisation check can rule the dataset out."""
    assert parse_tf_from_group_labels("group: ch1_VPH1 field: position") == ["Vph1"]


# ---------------------------------------------------------------------------
# The enumerating sources short-circuit
# ---------------------------------------------------------------------------

def test_group_labels_win_outright():
    """A group label lists one TF per chamber, so nothing is merged in."""
    text = log("Strain: Msn2-GFP Comments:") + "group: ch1_REI1 field: pos001\n"
    parsed = parse_tf_from_groups(text, "Msn2_screen_00")
    assert parsed.value == ["Rei1"]
    assert parsed.label() == "group-labels"


def test_strain_database_wins_outright():
    """Group 898 is curated as Msn2 + Mig1; the details must not add to it."""
    text = log("Strain: Dot6-GFP Comments:")
    parsed = parse_tf_from_groups(text, "Dot6_00", ["898"], {"898": ["Msn2", "Mig1"]})
    assert parsed.value == ["Msn2", "Mig1"]
    assert parsed.label() == "strain-db"


# ---------------------------------------------------------------------------
# The three free-text sources merge
# ---------------------------------------------------------------------------

def test_details_and_omero_tags_are_merged():
    """Dataset 831: the Strain: field truncates 'Sfp1-GFP/', the tag line does not."""
    text = log(
        "Aim: Alan's first glucose stress experiment. "
        "Strain: Msn2-GFP/Dof6=mCherry, Dof6-GFP/Msn2-mCherry, Sfp1-GFP/, wild type (By4741) Comments:",
        tail="Omero tags:\nBatgirl,Alan,mCherry,GFP,Sfp1-GFP/Mig1-mCherry,0.1% glucose,",
    )
    parsed = parse_tf_from_groups(text, "GlucoseStress_Alan_03")
    assert parsed.value == ["Msn2", "Sfp1", "Mig1"]
    assert parsed.label() == "details,tagged-proteins"


def test_details_lead_the_merged_result():
    """The experimenter's own strain list comes first in the column."""
    text = log("Strain: Hog1-GFP Comments:", tail="Omero tags:\nBatgirl,Msn2-GFP,")
    parsed = parse_tf_from_groups(text, "Dot6_ramp_00")
    assert parsed.value == ["Hog1", "Dot6", "Msn2"]
    assert parsed.label() == "details,dataset-name,tagged-proteins"


def test_dataset_name_still_read_when_details_are_silent():
    text = log("Aim: Image of graticule slide  Strain:   Comments:")
    parsed = parse_tf_from_groups(text, "Ramp_2to0pGlc_Msn2Dot6Mig1_00")
    assert parsed.value == ["Msn2", "Dot6", "Mig1"]
    assert parsed.label() == "dataset-name"


def test_double_colon_allele_notation_is_a_tagged_protein():
    """'htb2::mCherry' is the same construct as 'Htb2-mCherry'."""
    text = log("Aim: repeat  Strain:   Comments:", tail="YST1522 (msn2::mCherry)")
    assert parse_tf_from_groups(text, "flavin_00").value == ["Msn2"]


def test_nothing_identifiable_gives_unknown():
    """With no client the LLM fallback is skipped and UNKNOWN is returned."""
    text = log("Aim: Image of graticule slide for calibration  Strain:   Comments:")
    parsed = parse_tf_from_groups(text, "10xgraticule_00", client=None)
    assert parsed.value == ["UNKNOWN"]
    assert parsed.label() == "none"
