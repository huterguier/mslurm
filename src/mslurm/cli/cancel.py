"""mslurm cancel: scancel jobs given as cluster:jobid."""

from __future__ import annotations

import shlex

from mslurm.cli.common import group_by_cluster, load_config, resolve_refs
from mslurm.ssh import Remote


def add_parser(sub):
    p = sub.add_parser("cancel", help="cancel jobs (cluster:jobid ...)")
    p.add_argument("jobs", nargs="+", metavar="JOB")
    p.add_argument("-M", "--cluster", help="cluster for bare job ids")
    p.set_defaults(run=run)


def run(args) -> int:
    cfg = load_config()
    refs = resolve_refs(cfg, args.jobs, args.cluster)
    status = 0
    for name, group in group_by_cluster(refs).items():
        ids = [r.job_id for r in group]
        proc = Remote(cfg.get(name)).run("scancel -v " + " ".join(map(shlex.quote, ids)), check=False)
        if proc.returncode != 0:
            print(f"mslurm: {name}: {proc.stderr.strip()}")
            status = 1
        else:
            print("cancelled " + " ".join(str(r) for r in group))
    return status
