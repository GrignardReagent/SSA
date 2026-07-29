#!/usr/bin/python3

import time
import wandb

def init_wandb_run(config: dict):
    """
    config must contain:
        - project
        - entity
        - run_name

    All other keys are treated as wandb hyperparameters.
    """
    entity = config.get("entity")
    project = config.get("project")
    name = config.get("name")

    if project is None or entity is None or name is None:
        raise ValueError("wandb_config must include 'project', 'entity', and 'name'.")

    # shallow copy to avoid mutating caller's dict
    config_for_wandb = dict(config)
    del config_for_wandb["entity"]
    del config_for_wandb["project"]
    del config_for_wandb["name"]

    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config_for_wandb,
    )


def wandb_log(run, metrics: dict):
    run.log(metrics)


def finish_wandb_run(run, best_val_acc, start_time, finish: bool = True):
    """Write the end-of-training summary values, and by default close the run.

    Pass ``finish=False`` to leave the run open so the caller can add its own
    summary values first -- e.g. one-shot final test metrics computed after
    training returns, which would otherwise arrive too late to be logged.
    The caller is then responsible for calling ``run.finish()``.
    """
    elapsed = time.time() - start_time
    run.summary["training_time_sec"] = elapsed
    run.summary["best_val_acc"] = best_val_acc
    if finish:
        run.finish()

