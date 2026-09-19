"""mslurm show: everything known about one job, local and from Slurm."""

from __future__ import annotations

import shlex

from mslurm import index
from mslurm.cli.common import find_workdir, load_config, read_meta, resolve_refs
from mslurm.ssh import Remote

SACCT_DETAIL = "JobID,JobName,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,Partition,NodeList,AllocTRES,WorkDir,Reason"


def add_parser(sub):
    p = sub.add_parser("show", help="details for one job")
    p.add_argument("job", metavar="JOB")
    p.add_argument("-M", "--cluster", help="cluster for a bare job id")
    p.set_defaults(run=run)


def run(args) -> int:
    cfg = load_config()
    (ref,) = resolve_refs(cfg, [args.job], args.cluster)
    remote = Remote(cfg.get(ref.cluster))

    rec = index.get(ref.cluster, ref.job_id)
    workdir = rec.workdir if rec else None
    if workdir is None:
        try:
            workdir = find_workdir(remote, ref)
        except Exception:
            workdir = None
    meta = read_meta(remote, workdir) if workdir else {}

    info = {
        "job": str(ref),
        "name": (rec and rec.name) or meta.get("name"),
        "command": (rec and rec.command) or meta.get("command"),
        "code": (rec and rec.code) or meta.get("code"),
        "head": (rec and rec.head) or meta.get("head"),
        "dirty": (rec.dirty if rec else meta.get("dirty")),
        "submitted from": (rec and f"{rec.local_root}/{rec.subdir}".rstrip("/")) or meta.get("local_root"),
        "submitted": (rec and rec.submitted) or meta.get("submitted"),
        "workdir": workdir,
    }
    width = max(len(k) for k in info)
    for key, value in info.items():
        if value not in (None, ""):
            print(f"{key.ljust(width)}  {value}")
    print()

    jid = shlex.quote(ref.job_id)
    script = f"scontrol show job {jid} 2>/dev/null || sacct -j {jid} -X -P -o {SACCT_DETAIL}"
    proc = remote.run(script, check=False)
    print(proc.stdout.rstrip())
    if proc.returncode != 0 and proc.stderr.strip():
        print(proc.stderr.strip())
    return proc.returncode
