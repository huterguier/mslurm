"""Job references: `cluster:jobid`, where jobid may be an array task like 123_4."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mslurm import MslurmError

# 123, 123_4, 123_[0-9], 123_[0-9%2], 123+0 (het jobs)
JOB_ID_RE = re.compile(r"^\d+(?:_\d+|_\[[^\]]+\]|\+\d+)?$")


@dataclass(frozen=True, order=True)
class JobRef:
    cluster: str
    job_id: str

    def __str__(self) -> str:
        return f"{self.cluster}:{self.job_id}"

    @property
    def base(self) -> str:
        """The array master / plain job id: 123 for 123_4."""
        return re.split(r"[_+]", self.job_id, maxsplit=1)[0]

    @property
    def is_array_task(self) -> bool:
        return "_" in self.job_id and "[" not in self.job_id


def parse_ref(text: str) -> tuple[str | None, str]:
    """Split `cluster:jobid` or bare `jobid` into (cluster or None, jobid)."""
    cluster, sep, job_id = text.partition(":")
    if not sep:
        cluster, job_id = None, text
    if not JOB_ID_RE.match(job_id):
        raise MslurmError(f"'{text}' is not a job id (expected cluster:jobid, e.g. alpha:12345 or alpha:12345_3)")
    return (cluster or None), job_id


def output_name(job_id: str, stderr: bool = False) -> str:
    """Slurm's default output file name for a job id: slurm-123.out, slurm-123_4.out."""
    return f"slurm-{job_id}.{'err' if stderr else 'out'}"
