"""mslurm ssh: an interactive shell on a cluster, sharing the connection other commands reuse."""

from __future__ import annotations

from mslurm.cli.common import load_config
from mslurm.ssh import Remote


def add_parser(sub):
    p = sub.add_parser("ssh", help="open a shell on a cluster (or run `-- CMD` there)")
    p.add_argument("cluster", nargs="?")
    p.set_defaults(run=run)


def run(args) -> int:
    cfg = load_config()
    Remote(cfg.pick(args.cluster)).exec_shell(args.rest or None)
    return 0
