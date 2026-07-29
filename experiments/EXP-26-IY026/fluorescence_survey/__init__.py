"""EXP-26-IY026: survey of the lab OMERO server for fluorescence time-lapse experiments.

Module map (see ../README.md for the full walkthrough):
    config           run settings and credentials, all overridable from .env
    omero_source     OMERO connection + reading log/Acq text annotations
    vocabulary       known fluorescence channels, TFs and non-TF markers
    strain_db        strain_tf_database.csv lookup
    parse_channels   channel names from the four metadata layouts
    parse_experiment timepoints, strain IDs, experimental condition
    parse_tf         transcription-factor identification
    llm              OpenAI client + structured-reply helpers
    classification   the two per-dataset verdicts and the results.csv row
    pipeline         the run loop over all datasets
"""
