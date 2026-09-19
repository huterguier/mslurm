"""mslurm info: partitions and GPUs on every cluster."""

from __future__ import annotations

from mslurm.cli.common import each_cluster, load_config
from mslurm.slurm import SINFO_FORMAT, parse_sinfo
from mslurm.ssh import Remote
from mslurm.table import print_table


def add_parser(sub):
    p = sub.add_parser("info", help="partitions, nodes and GPUs per cluster")
    p.add_argument("-M", "--cluster", action="append", help="only this cluster (repeatable)")
    p.set_defaults(run=run)


def run(args) -> int:
    cfg = load_config()
    clusters = cfg.select(args.cluster)
    results = each_cluster(clusters, lambda c: parse_sinfo(Remote(c).run(f"sinfo -h -N -o {SINFO_FORMAT!r}").stdout))

    rows = []
    for cluster in clusters:
        for p in sorted(results.get(cluster.name, []), key=lambda p: (not p.default, p.name)):
            gpus = ", ".join(f"{g} {p.gpus_idle.get(g, 0)}/{n}" for g, n in sorted(p.gpus.items()))
            rows.append([cluster.name, p.name + ("*" if p.default else ""), p.avail, p.limit, f"{p.idle}/{p.nodes}", gpus])
    print_table(["CLUSTER", "PARTITION", "AVAIL", "TIMELIMIT", "NODES idle/all", "GPUS idle/all"], rows, max_width=80)

    aliases = [(c.name, a, g) for c in clusters for a, g in c.gpus.items()]
    if aliases:
        print("\ngpu aliases: " + ", ".join(f"{c}:{a}={g.type}" + (f"@{g.partition}" if g.partition else "") for c, a, g in aliases))
    return 0
