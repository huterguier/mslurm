"""Turn a cluster profile plus command-line resources into an sbatch script."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from mslurm import MslurmError
from mslurm.config import Cluster
from mslurm.slurm import slurm_time
from mslurm.ssh import sh_path


@dataclass
class Resources:
    name: str | None = None
    time: str | None = None
    gpus: str | None = None  # "2", "rtx3090", "rtx3090:2" or a configured alias
    partition: str | None = None
    cpus: str | None = None
    mem: str | None = None
    nodes: str | None = None
    ntasks: str | None = None
    array: str | None = None
    account: str | None = None
    qos: str | None = None
    extra: list[str] = field(default_factory=list)  # raw sbatch flags, e.g. --exclusive
    env: dict[str, str] = field(default_factory=dict)


def resolve_gpus(cluster: Cluster, spec: str | None) -> tuple[str | None, str | None]:
    """`-G` value -> (gres string, partition override from the alias, if any)."""
    if spec is None or spec in ("", "0", "none"):
        return None, None
    name, _, count = spec.partition(":")
    if count and not count.isdigit():
        raise MslurmError(f"bad --gpus value '{spec}' (use N, TYPE or TYPE:N)")
    alias = cluster.gpus.get(name)
    if alias is None and spec.isdigit():
        return f"gpu:{spec}", None
    gpu_type = alias.type if alias else name
    partition = alias.partition if alias else None
    return f"gpu:{gpu_type}:{count or 1}", partition


_SHORT_DIRECTIVES = {
    "-A": "account", "-q": "qos", "-p": "partition", "-J": "job-name", "-t": "time", "-c": "cpus-per-task",
    "-n": "ntasks", "-N": "nodes", "-a": "array", "-G": "gpus", "-o": "output", "-e": "error", "-M": "clusters",
}


def sbatch_directives(script: str) -> set[str]:
    """Long option names set by a script's #SBATCH lines (`-A x`, `--account=x` and `--account x`)."""
    found: set[str] = set()
    for line in script.splitlines():
        line = line.strip()
        if not line.startswith("#SBATCH"):
            continue
        try:
            tokens = shlex.split(line[len("#SBATCH"):], comments=True)
        except ValueError:
            continue
        for tok in tokens:
            if tok.startswith("--"):
                found.add(tok[2:].split("=", 1)[0])
            elif tok.startswith("-") and len(tok) >= 2 and tok[:2] in _SHORT_DIRECTIVES:
                found.add(_SHORT_DIRECTIVES[tok[:2]])
    return found


def sbatch_options(
    cluster: Cluster,
    res: Resources,
    comment: str | None = None,
    *,
    script_mode: bool = False,
    script_has: set[str] = frozenset(),
) -> list[tuple[str, str | None]]:
    """Ordered sbatch long options. `None` values are flags without an argument.

    In script mode the user's own #SBATCH directives should win, so cluster
    defaults, the default partition and the output name are left out, and the
    profile's account and qos are only added when the script does not set them.
    Explicit command-line values still override the script, as with sbatch.
    """
    gres, gres_partition = resolve_gpus(cluster, res.gpus)
    opts: dict[str, str | None] = {}
    if res.name:
        opts["job-name"] = slug(res.name, 64)

    partition = res.partition or gres_partition or (None if script_mode else cluster.partition)
    account = res.account or (None if "account" in script_has else cluster.account)
    qos = res.qos or (None if "qos" in script_has else cluster.qos)
    if account:
        opts["account"] = account
    if qos:
        opts["qos"] = qos
    if partition:
        opts["partition"] = partition

    given = {
        "time": slurm_time(res.time) if res.time else None,
        "gres": gres,
        "cpus-per-task": res.cpus,
        "mem": res.mem,
        "nodes": res.nodes,
        "ntasks": res.ntasks,
        "array": res.array,
    }
    if not script_mode:
        for key, value in cluster.defaults.items():
            if key not in opts and given.get(key) is None:
                opts[key] = slurm_time(value) if key == "time" else value
    for key, value in given.items():
        if value is not None:
            opts[key] = value

    if not script_mode:
        opts["output"] = "slurm-%A_%a.out" if res.array else "slurm-%j.out"
    if comment:
        opts["comment"] = comment
    result = list(opts.items())
    for raw in res.extra:
        flag, sep, value = raw.lstrip("-").partition("=")
        result.append((flag, value if sep else None))
    return result


def sbatch_args(options: list[tuple[str, str | None]]) -> list[str]:
    return [f"--{k}" if v is None else f"--{k}={v}" for k, v in options]


def render_script(cluster: Cluster, res: Resources, command: list[str], *, code: str, comment: str | None = None) -> str:
    lines = ["#!/bin/bash"]
    lines += [f"#SBATCH {arg}" for arg in sbatch_args(sbatch_options(cluster, res, comment))]
    lines += [
        "",
        "set -euo pipefail",
        f"export MSLURM_CODE={sh_path(code)}",
        'export MSLURM_JOB_DIR="$PWD"',
    ]
    env = {**cluster.env, **res.env}
    lines += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    if cluster.setup:
        lines += ["", "# setup (from clusters.yaml)", cluster.setup]
    lines += ["", shlex.join(command)]
    return "\n".join(lines) + "\n"


_INTERPRETERS = {"python", "python3", "uv", "run", "bash", "sh", "srun", "env", "poetry", "pdm", "torchrun", "accelerate", "launch"}


def guess_name(command: list[str], fallback: str) -> str:
    """A job name from the command: `uv run train.py --lr 1e-3` -> train."""
    tokens = list(command)
    while tokens:
        tok = tokens.pop(0)
        if tok in _INTERPRETERS or "=" in tok and not tok.startswith("-") and tokens:
            continue
        if tok == "-m" and tokens:
            return slug(tokens[0].split(".")[-1])
        if tok.startswith("-"):
            continue
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", tok.rsplit("/", 1)[-1])
        return slug(stem) if stem else slug(fallback)
    return slug(fallback)


def slug(text: str, limit: int = 40) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return (text or "job")[:limit]
