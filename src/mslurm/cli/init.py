"""mslurm init: create ~/.config/mslurm/clusters.yaml from the example."""

from __future__ import annotations

import os
import shutil

from mslurm import MslurmError
from mslurm.config import CONFIG_PATH, EXAMPLE_PATH


def add_parser(sub):
    p = sub.add_parser("init", help="write an example clusters.yaml to edit")
    p.add_argument("--force", action="store_true", help="overwrite an existing config")
    p.set_defaults(run=run)


def run(args) -> int:
    if os.path.exists(CONFIG_PATH) and not args.force:
        raise MslurmError(f"{CONFIG_PATH} already exists (use --force to overwrite)")
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    shutil.copy(EXAMPLE_PATH, CONFIG_PATH)
    print(f"wrote {CONFIG_PATH}; edit the host, account and setup lines for your clusters")
    return 0
