"""Helpers shared by the subcommands: resolving job references and remote job dirs."""

from __future__ import annotations

import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from mslurm import MslurmError, config, index
from mslurm.config import Cluster, Config
from mslurm.ids import JobRef, parse_ref
from mslurm.ssh import Remote, RemoteError, sh_path

T = TypeVar("T")


def warn(msg: str) -> None:
    print(f"mslurm: {msg}", file=sys.stderr)


def resolve_refs(cfg: Config, texts: list[str], cluster_flag: str | None = None) -> list[JobRef]:
    """`alpha:123` as given; a bare `123` via -M, the local index, or the only configured cluster.

    A bare id never falls back to the default cluster: with two clusters that
    would make `cancel 12345` a guess, and a wrong guess kills someone's job.
    """
    refs = []
    for text in texts:
        cluster, job_id = parse_ref(text)
        if cluster is None:
            cluster = cluster_flag
        if cluster is None:
            known = index.clusters_for(job_id)
            if len(known) == 1:
                cluster = known[0]
            elif len(known) > 1:
                raise MslurmError(f"job {job_id} exists on {', '.join(known)}; say which with cluster:{job_id}")
        if cluster is None and len(cfg.clusters) == 1:
            cluster = next(iter(cfg.clusters))
        if cluster is None:
            raise MslurmError(f"job {job_id} is not in the local index; say which cluster with cluster:{job_id} or -M")
        cfg.get(cluster)  # validate
        refs.append(JobRef(cluster, job_id))
    return refs


def group_by_cluster(refs: list[JobRef]) -> dict[str, list[JobRef]]:
    groups: dict[str, list[JobRef]] = {}
    for ref in refs:
        groups.setdefault(ref.cluster, []).append(ref)
    return groups


def find_workdir(remote: Remote, ref: JobRef) -> str:
    """Absolute working directory of a job: from the local index, else from Slurm."""
    rec = index.get(ref.cluster, ref.job_id)
    if rec and rec.workdir:
        return rec.workdir
    jid = shlex.quote(ref.job_id)
    script = (
        f"d=$(squeue -h -j {jid} -o %Z 2>/dev/null | head -1); "
        f'[ -n "$d" ] || d=$(sacct -X -n -P -j {jid} -o WorkDir 2>/dev/null | head -1); '
        'echo "$d"'
    )
    workdir = remote.run(script).stdout.strip()
    if not workdir:
        raise MslurmError(f"{ref}: Slurm has no record of this job (and it is not in the local index)")
    return workdir


def read_meta(remote: Remote, workdir: str) -> dict:
    """The mslurm.json a submit wrote into the job dir, or {} for jobs not submitted by mslurm."""
    import json

    # The job's workdir may be a subdirectory of the job dir, which holds mslurm.json at its top.
    d = sh_path(workdir)
    script = f'for d in {d} {d}/.. {d}/../.. {d}/../../..; do [ -f "$d/mslurm.json" ] && cat "$d/mslurm.json" && exit 0; done; exit 1'
    proc = remote.run(script, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def each_cluster(clusters: list[Cluster], fn: Callable[[Cluster], T]) -> dict[str, T]:
    """Run fn on every cluster in parallel; report failures on stderr and return the successes."""
    results: dict[str, T] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(clusters))) as pool:
        futures = {pool.submit(fn, c): c for c in clusters}
        for future, cluster in futures.items():
            try:
                results[cluster.name] = future.result()
            except RemoteError as exc:
                warn(str(exc).splitlines()[0])
    return results


def load_config() -> Config:
    return config.load()
