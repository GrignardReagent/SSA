"""Controlled vocabularies: fluorescence channels and protein classifications.

These sets are the ground truth used by the deterministic parsers, so that the
LLM is only consulted when a name is not recognised here.
"""

# Channel names that count as fluorescence acquisitions (brightfield does not)
FLUORESCENCE_CHANNELS = (
    "GFP",
    "GFP_Z",
    "GFPFast",
    "mCherry",
    "mCherry2",
    "mCherry_Z",
    "Citrine",
    "Flavin",
    "mKO2",
    "Cy5",
    "cy5",
    "pHluorin405",
    "pHluorin488",
    "NADH",
    "mTurquoise2",
    "tdTomatoFRET",
    "YFP",       # dataset 932: DIC + YFP over 200 timepoints
    "TMRM",      # mitochondrial membrane-potential dye (datasets 4418-4420, 3709)
    "coumarin",  # NADH-experiment dye (datasets 1239, 2709, 2711)
)

# Transmitted-light channels. Named explicitly rather than treated as "anything
# not fluorescent", so a channel name nobody has catalogued yet stands out as
# unrecognised instead of being silently filed as brightfield.
BRIGHTFIELD_CHANNELS = ("Brightfield", "brightfield1", "DIC")

# All proteins confirmed as TF/nuclear-localisation reporters in Swain-lab experiments.
# Drawn from IY008 training metadata (143 confirmed TFs) plus extras seen in IY026.
KNOWN_TFS: frozenset[str] = frozenset({
    # IY008 confirmed TF localisation training data
    "Abf1", "Abf2", "Abf3", "Ace2", "Adr1", "Afg2", "Arg80", "Arg81", "Aro80",
    "Asg1", "Ash1", "Azf1", "Bas1", "Cbf1", "Cha4", "Cin5", "Cst6", "Cup2", "Cup9",
    "Dal81", "Def1", "Dig1", "Ecm22", "Ert1", "Fhl1", "Fkh1", "Fkh2", "Fzf1",
    "Gcn4", "Gcr1", "Gis1", "Gis4", "Gsh1", "Gts1", "Gzf1", "Gzf3", "Hac1", "Hal9",
    "Hap2", "Hap3", "Hap5", "Hcm1", "Hcs1", "Hsc82", "Ino2", "Ino4", "Ixr1",
    "Leu3", "Lys14", "Mac1", "Mac10", "Mbp1", "Mcm1", "Met28", "Met31", "Met32",
    "Met4", "Mig3", "Mip6", "Mot2", "Mot3", "Msn1", "Msn2", "Ndd1", "Nhp10",
    "Nhp6B", "Opi1", "Pan3", "Pbs2", "Pdr1", "Pdr3", "Pdr8", "Phd1", "Pho2", "Pip2",
    "Rbh1", "Rds2", "Reb1", "Rei1", "Rfx1", "Rgm1", "Rgt1", "Rlm1", "Rme1", "Rox1",
    "Rph1", "Rpn4", "Rsc3", "Rsf2", "Rxf1", "Sac7", "Sch9", "Sfa1", "Sfl1", "Skn7",
    "Snt2", "Sok2", "Spt15", "Spt2", "Srd1", "Ssa1", "Ssk1", "Sst2", "Stb1", "Stb3",
    "Stb5", "Ste12", "Stp1", "Stp2", "Stp3", "Stp4", "Sum1", "Sut1", "Swi4", "Swi5",
    "Swi6", "Tbf1", "Tbs1", "Tea1", "Tec1", "Tho2", "Ths1", "Tye7", "Uga3", "Ume1",
    "Ume3", "Ume6", "Upc2", "Whi3", "Xbp1", "Yap3", "Yap5", "Yap7", "Yhp1", "Yox1",
    "Ypk1", "Yrm1", "Yrr1", "Zap1",
    # Additional TFs / nuclear-localisation reporters seen in IY026 data
    "Mig1", "Mig2", "Msn4", "Dot6", "Sfp1", "Hog1", "Crz1", "Yap1", "Whi5",
    "Cat8", "Nrg1", "Rtg1", "Maf1", "Tod6", "Srl1", "Gln3", "Pho4",
    "Gal3", "Gal4", "Gal7", "Gal80",
    "Snf1", "Bcy1", "Yak1",   # signaling proteins with documented localisation dynamics
    "Bub1", "Mad1",            # kinetochore proteins tracked for localisation
})

# Fluorescently-tagged proteins that are NOT TFs / localisation reporters.
# Used to downgrade experiments that image only non-TF markers.
KNOWN_NON_TF_MARKERS: frozenset[str] = frozenset({
    # Histones / chromatin (mark nucleus but are not TFs)
    "Htb2", "Htb1", "Hta2", "Hta1",
    # Organelle markers
    "Vph1",   # vacuolar H⁺-ATPase
    "Cox4", "Cox5",  # mitochondria
    "Pex14",  # peroxisome
    # Bud-neck / SPB / cell-cycle structural proteins
    "Bud3", "Lte1", "Cdc14",
    # RNA-binding / stress-granule proteins
    "Pab1", "Pub1", "Ngr1", "Edc1",
    # Metabolic enzymes from protein-aggregation screens (non-TF)
    "Ade6", "Vip1", "Rse1", "Iki3", "Caf130", "Gdh2", "She4", "Amd1", "Pat1",
    "Gln1", "Elp2", "Ltv1", "Sro9", "Cpr6", "Hsp42", "Cdc123", "Scd6", "Tmt1",
    "Sbp1", "Nmd4", "Met14", "Lsm1", "Fas2", "Slh1", "Aro1", "Kem1", "Ura7",
    # Signalling (kinases / GEFs) that are clearly not localisation reporters
    "Sip1", "Sip2", "Gal83",  # Snf1 complex components (tracked as degradation targets)
})

# Lower-cased lookup tables, built once at import for fast membership tests
TF_LOOKUP: dict[str, str] = {tf.lower(): tf for tf in KNOWN_TFS}
NON_TF_LOOKUP: dict[str, str] = {m.lower(): m for m in KNOWN_NON_TF_MARKERS}
