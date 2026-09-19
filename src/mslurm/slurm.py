"""Pure parsers for squeue / sacct / sinfo output, and small Slurm value helpers.

Nothing here touches the network, so all of it is unit-testable with captured text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mslurm import MslurmError

SQUEUE_FORMAT = "%i|%j|%T|%P|%M|%l|%D|%N|%R|%Z|%b|%V"
SACCT_FIELDS = "JobID,JobName,State,Partition,Elapsed,Timelimit,NNodes,NodeList,Reason,WorkDir,AllocTRES,Submit,End"
SINFO_FORMAT = "%P|%a|%l|%t|%G"

ACTIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING", "STAGE_OUT"}
_STATE_ORDER = {"RUNNING": 0, "COMPLETING": 0, "CONFIGURING": 0, "PENDING": 1, "SUSPENDED": 1, "REQUEUED": 1}


@dataclass
class Job:
    id: str
    name: str
    state: str
    partition: str
    elapsed: str = ""
    limit: str = ""
    nodes: str = ""
    nodelist: str = ""
    reason: str = ""
    workdir: str = ""
    gpus: str = ""
    submit: str = ""
    end: str = ""

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def where(self) -> str:
        """Node list while running, the pending reason otherwise."""
        if self.state == "PENDING":
            reason = self.reason.strip("()")
            return f"({reason})" if reason and reason != "None" else ""
        return self.nodelist if self.nodelist not in ("", "None assigned") else ""


def parse_gpus(tres: str) -> str:
    """Shorten a gres/TRES string to what matters: `rtx3090:2`, `gpu:1` or ``."""
    if not tres or tres in ("N/A", "(null)"):
        return ""
    found = []
    for part in tres.split(","):
        part = part.strip()
        # squeue %b: gres/gpu:rtx3090:2 or gres/gpu:2 ; sacct AllocTRES: gres/gpu:rtx3090=2, gres/gpu=2
        if not part.startswith("gres/gpu"):
            continue
        pieces = [p for p in re.split(r"[:=]", part[len("gres/gpu"):]) if p]
        count = pieces.pop() if pieces and pieces[-1].isdigit() else "1"
        found.append((pieces[0] if pieces else None, count))
    typed = [f"{t}:{n}" for t, n in found if t]
    if typed:
        return ",".join(typed)
    untyped = [n for t, n in found if not t]
    return f"gpu:{untyped[0]}" if untyped else ""


def parse_gres_types(gres: str) -> dict[str, int]:
    """sinfo %G, e.g. `gpu:rtx3090:8(S:0-1),gpu:a6000:4` -> {rtx3090: 8, a6000: 4}."""
    out: dict[str, int] = {}
    if not gres or gres == "(null)":
        return out
    for part in gres.split(","):
        m = re.match(r"^gpu(?::([^:(]+))?:(\d+)", part.strip())
        if m:
            name = m.group(1) or "gpu"
            out[name] = out.get(name, 0) + int(m.group(2))
    return out


def parse_squeue(text: str) -> list[Job]:
    jobs = []
    for line in text.splitlines():
        cols = line.rstrip("\n").split("|")
        if len(cols) != 12:
            continue
        jid, name, state, part, elapsed, limit, nodes, nodelist, reason, workdir, gres, submit = cols
        jobs.append(Job(jid, name, state, part, elapsed, limit, nodes, nodelist, reason, workdir, parse_gpus(gres), submit))
    return jobs


def parse_sacct(text: str) -> list[Job]:
    jobs = []
    for line in text.splitlines():
        cols = line.rstrip("\n").split("|")
        if len(cols) != 13:
            continue
        jid, name, state, part, elapsed, limit, nodes, nodelist, reason, workdir, tres, submit, end = cols
        state = state.split()[0] if state else ""  # "CANCELLED by 1000" -> CANCELLED
        jobs.append(Job(jid, name, state, part, elapsed, limit, nodes, nodelist, reason, workdir, parse_gpus(tres), submit, end))
    return jobs


def merge_jobs(live: list[Job], history: list[Job]) -> list[Job]:
    """squeue rows win over sacct rows for the same id; result is sorted running, pending, then newest first."""
    by_id: dict[str, Job] = {}
    for job in history:
        by_id[job.id] = job
    for job in live:
        by_id[job.id] = job

    def key(job: Job):
        return (_STATE_ORDER.get(job.state, 2), -_numeric(job.id))

    return sorted(by_id.values(), key=key)


def _numeric(job_id: str) -> float:
    m = re.match(r"^(\d+)(?:_(\d+))?", job_id)
    if not m:
        return 0
    return int(m.group(1)) + (int(m.group(2)) / 1e6 if m.group(2) else 0)


@dataclass
class Partition:
    name: str
    default: bool = False
    avail: str = ""
    limit: str = ""
    nodes: int = 0
    idle: int = 0
    gpus: dict[str, int] = field(default_factory=dict)
    gpus_idle: dict[str, int] = field(default_factory=dict)


def parse_sinfo(text: str) -> list[Partition]:
    """Aggregate `sinfo -N` rows (one per node and partition) into per-partition totals."""
    parts: dict[str, Partition] = {}
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) != 5:
            continue
        pname, avail, limit, state, gres = cols
        default = pname.endswith("*")
        pname = pname.rstrip("*")
        p = parts.setdefault(pname, Partition(pname, default, avail, limit))
        p.nodes += 1
        idle = state.rstrip("*~#!%$@^-").lower() == "idle"
        if idle:
            p.idle += 1
        for gpu, count in parse_gres_types(gres).items():
            p.gpus[gpu] = p.gpus.get(gpu, 0) + count
            if idle:
                p.gpus_idle[gpu] = p.gpus_idle.get(gpu, 0) + count
    return list(parts.values())


_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(m|min|h|hr|d|w)$")
_UNIT_SECONDS = {"m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400, "w": 604800}


def parse_duration(spec: str) -> int:
    """`30m`, `2h`, `1.5h`, `3d`, `1w` -> seconds. Slurm-style values raise."""
    m = _DURATION_RE.match(spec.strip().lower())
    if not m:
        raise MslurmError(f"cannot parse duration '{spec}' (use e.g. 30m, 2h, 3d, 1w)")
    return int(float(m.group(1)) * _UNIT_SECONDS[m.group(2)])


def slurm_time(spec: str) -> str:
    """Accept `5m`/`2h`/`1d` shorthands and pass Slurm's own formats through."""
    if ":" in spec or "-" in spec or spec.isdigit():
        return spec
    total = parse_duration(spec)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def sacct_since(spec: str) -> str:
    """`1d` -> `now-1days`, the form sacct --starttime understands."""
    total = parse_duration(spec)
    if total % 86400 == 0:
        return f"now-{total // 86400}days"
    if total % 3600 == 0:
        return f"now-{total // 3600}hours"
    return f"now-{max(1, total // 60)}minutes"
