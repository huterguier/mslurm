"""mslurm clean: delete finished job directories and code snapshots nothing refers to."""

from __future__ import annotations

import sys

from mslurm import MslurmError
from mslurm.cli.common import load_config
from mslurm.slurm import parse_duration
from mslurm.ssh import Remote, sh_path
from mslurm.table import print_table


def add_parser(sub):
    p = sub.add_parser("clean", help="remove old job dirs and unused code snapshots on a cluster")
    p.add_argument("-M", "--cluster", action="append", help="only this cluster (repeatable)")
    p.add_argument("--older-than", default="14d", help="job dirs with nothing modified in this long (default 14d)")
    p.add_argument("--dry-run", action="store_true", help="list only")
    p.add_argument("-y", "--yes", action="store_true", help="delete without asking")
    p.set_defaults(run=run)


def _script(cluster, days: int, delete: bool) -> str:
    rm = "rm -rf --" if delete else ":"
    rm_tmp = f"find \"$code\" -maxdepth 1 -name '.tmp-*' -mtime +1 -exec rm -rf -- {{}} + 2>/dev/null" if delete else ":"
    return f"""
jobs={sh_path(cluster.path("jobs"))}
code={sh_path(cluster.path("code"))}
active=$(squeue --me -h -o %Z 2>/dev/null | sort -u)
is_active() {{ printf '%s\\n' "$active" | awk -v d="$1" '$0 == d || index($0, d "/") == 1 {{ f = 1 }} END {{ exit !f }}'; }}
if [ -d "$jobs" ]; then
  for d in "$jobs"/*/; do
    [ -d "$d" ] || continue
    d=${{d%/}}
    is_active "$d" && continue
    [ -z "$(find "$d" -mtime -{days} -print -quit 2>/dev/null)" ] || continue
    echo "JOB $(basename "$d") $(du -sk "$d" | cut -f1)"
    {rm} "$d"
  done
fi
if [ -d "$code" ]; then
  referenced=$(find "$jobs" -maxdepth 2 -name mslurm.json -exec grep -ho '"code": *"[0-9a-f]*"' {{}} + 2>/dev/null | grep -o '[0-9a-f]\\{{40\\}}' | sort -u)
  for s in "$code"/*/; do
    [ -d "$s" ] || continue
    s=${{s%/}}
    sha=$(basename "$s")
    printf '%s\\n' "$referenced" | grep -qxF -- "$sha" && continue
    [ -z "$(find "$s" -maxdepth 0 -mtime -1)" ] || continue
    echo "CODE $sha $(du -sk "$s" | cut -f1)"
    {rm} "$s"
  done
  {rm_tmp}
fi
true
"""


def _plan(cluster, days: int) -> list[tuple[str, str, int]]:
    out = Remote(cluster).run(_script(cluster, days, delete=False)).stdout
    items = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in ("JOB", "CODE"):
            items.append((parts[0], parts[1], int(parts[2])))
    return items


def run(args) -> int:
    cfg = load_config()
    days = max(0, parse_duration(args.older_than) // 86400)
    total_kb = 0
    plans = {}
    for cluster in cfg.select(args.cluster):
        items = _plan(cluster, days)
        plans[cluster.name] = items
        for kind, name, kb in items:
            total_kb += kb
        if items:
            print(f"{cluster.name}:")
            print_table(["", "PATH", "SIZE"], [[k.lower(), n, _human_kb(kb)] for k, n, kb in items], max_width=80)
    if not any(plans.values()):
        print("nothing to clean")
        return 0
    print(f"\n{sum(len(v) for v in plans.values())} items, {_human_kb(total_kb)}")
    if args.dry_run:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            raise MslurmError("refusing to delete without a terminal; pass -y")
        answer = input("delete? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return 1
    for cluster in cfg.select(args.cluster):
        if plans[cluster.name]:
            Remote(cluster).run(_script(cluster, days, delete=True))
            print(f"{cluster.name}: deleted {len(plans[cluster.name])} items")
    return 0


def _human_kb(kb: int) -> str:
    size = float(kb)
    for unit in ("KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
