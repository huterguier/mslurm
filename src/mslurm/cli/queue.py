"""mslurm queue: your jobs on every cluster, live from squeue plus recent history from sacct."""

from __future__ import annotations

from mslurm.cli.common import each_cluster, load_config
from mslurm.slurm import SACCT_FIELDS, SQUEUE_FORMAT, Job, merge_jobs, parse_sacct, parse_squeue, sacct_since
from mslurm.ssh import Remote
from mslurm.table import print_table

SEPARATOR = "__MSLURM_SACCT__"


def add_parser(sub):
    p = sub.add_parser("queue", aliases=["q"], help="list your jobs on all clusters")
    p.add_argument("-M", "--cluster", action="append", help="only this cluster (repeatable)")
    p.add_argument("--since", default="1d", help="how far back to show finished jobs (default 1d)")
    p.add_argument("--all", action="store_true", help="finished jobs from the last 30 days")
    p.add_argument("--active", action="store_true", help="only running and pending jobs")
    p.set_defaults(run=run)


def fetch(cluster, since: str) -> list[Job]:
    script = (
        f"squeue --me -h -o {SQUEUE_FORMAT!r}; echo {SEPARATOR}; "
        f"sacct -X -n -P --starttime {sacct_since(since)} -o {SACCT_FIELDS} 2>/dev/null || true"
    )
    out = Remote(cluster).run(script).stdout
    live, _, history = out.partition(SEPARATOR + "\n")
    return merge_jobs(parse_squeue(live), parse_sacct(history))


def run(args) -> int:
    cfg = load_config()
    clusters = cfg.select(args.cluster)
    since = "30d" if args.all else args.since
    results = each_cluster(clusters, lambda c: fetch(c, since))

    rows = []
    for cluster in clusters:
        for job in results.get(cluster.name, []):
            if args.active and not job.active:
                continue
            rows.append([
                f"{cluster.name}:{job.id}", job.name, job.state, job.partition, job.gpus,
                job.elapsed, job.limit, job.where,
            ])
    if not rows:
        print("no jobs" if not args.active else "no running or pending jobs")
        return 0
    print_table(["ID", "NAME", "STATE", "PARTITION", "GPUS", "ELAPSED", "LIMIT", "NODE/REASON"], rows)
    return 0
