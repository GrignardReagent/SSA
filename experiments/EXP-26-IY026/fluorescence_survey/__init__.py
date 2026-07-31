"""EXP-26-IY026: survey of the lab OMERO server for fluorescence time-lapse experiments.

Module map (see ../README.md for the full walkthrough):
    config           run settings and credentials, all overridable from .env
    omero_source     OMERO connection + reading log/Acq text annotations
    vocabulary       known channel names, TFs and non-TF markers
    strain_db        strain_tf_database.csv lookup
    utils            dedupe helper + `Parsed`, a value with its provenance
    parse_channels   channel names from the four metadata layouts
    parse_experiment the details block, timepoints, strain IDs, condition
    parse_tf         transcription-factor identification
    llm              OpenAI client, per-dataset call transcript, reply parsing
    classification   the two per-dataset verdicts and the results.csv row
    pipeline         the run loop over all datasets

Every field in results.csv is produced by a chain of fallbacks, so the parsers
return `Parsed` — the value plus the extractor(s) that produced it. That
provenance reaches the CSV, because how far to trust a value depends on which
link fired.
"""
