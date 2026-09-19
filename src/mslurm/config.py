"""Cluster profiles from ~/.config/mslurm/clusters.yaml.

One profile per cluster: how to reach it, where to put files, and the sbatch
settings every job on it needs. See data/clusters.example.yaml for the format.
"""

from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass, field

import yaml

from mslurm import MslurmError

CONFIG_DIR = os.environ.get("MSLURM_CONFIG_DIR") or os.path.expanduser("~/.config/mslurm")
CONFIG_PATH = os.path.join(CONFIG_DIR, "clusters.yaml")
CACHE_DIR = os.environ.get("MSLURM_CACHE_DIR") or os.path.expanduser("~/.cache/mslurm")
STATE_DIR = os.environ.get("MSLURM_STATE_DIR") or os.path.join(
    os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"), "mslurm"
)
EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "clusters.example.yaml")


class ConfigError(MslurmError):
    pass


@dataclass(frozen=True)
class Gpu:
    """A GPU alias: the gres type name Slurm knows, plus an optional partition it lives in."""

    type: str
    partition: str | None = None


@dataclass(frozen=True)
class Cluster:
    name: str
    host: str
    user: str | None = None
    scratch: str = "~/mslurm"
    account: str | None = None
    qos: str | None = None
    partition: str | None = None
    gpus: dict[str, Gpu] = field(default_factory=dict)
    setup: str = ""
    # sbatch long-option name -> value, applied when the flag isn't given on the command line.
    defaults: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def path(self, *parts: str) -> str:
        """A path under the scratch root. `~` is kept so the remote shell expands it."""
        return posixpath.join(self.scratch, *parts)


@dataclass
class Config:
    clusters: dict[str, Cluster]
    default: str | None = None

    def get(self, name: str) -> Cluster:
        try:
            return self.clusters[name]
        except KeyError:
            raise ConfigError(f"unknown cluster '{name}'. Configured: {', '.join(self.clusters) or '(none)'}")

    def pick(self, name: str | None = None) -> Cluster:
        """The cluster to use: an explicit name, else $MSLURM_CLUSTER, else the default."""
        name = name or os.environ.get("MSLURM_CLUSTER") or self.default
        if name is None and len(self.clusters) == 1:
            name = next(iter(self.clusters))
        if name is None:
            raise ConfigError("no cluster given: pass -M/--cluster or set `default:` in clusters.yaml")
        return self.get(name)

    def select(self, names: list[str] | None) -> list[Cluster]:
        """Clusters for a -M filter, or all of them."""
        if not names:
            return list(self.clusters.values())
        return [self.get(n) for n in names]


_CLUSTER_KEYS = {"host", "user", "scratch", "account", "qos", "partition", "gpus", "setup", "defaults", "env"}


def parse(raw: dict) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("clusters.yaml must be a mapping with a `clusters:` key")
    unknown = set(raw) - {"clusters", "default"}
    if unknown:
        raise ConfigError(f"unknown top-level keys: {', '.join(sorted(unknown))}")
    clusters = {}
    for name, spec in (raw.get("clusters") or {}).items():
        clusters[name] = _parse_cluster(str(name), spec or {})
    default = raw.get("default")
    if default is not None and default not in clusters:
        raise ConfigError(f"`default: {default}` does not name a configured cluster")
    return Config(clusters=clusters, default=default)


def _parse_cluster(name: str, spec: dict) -> Cluster:
    if not isinstance(spec, dict):
        raise ConfigError(f"cluster '{name}' must be a mapping")
    unknown = set(spec) - _CLUSTER_KEYS
    if unknown:
        raise ConfigError(f"cluster '{name}': unknown keys {', '.join(sorted(unknown))}")
    if not spec.get("host"):
        raise ConfigError(f"cluster '{name}' needs `host:` (an ssh host name or alias)")

    gpus = {}
    for alias, g in (spec.get("gpus") or {}).items():
        if isinstance(g, str):
            gpus[str(alias)] = Gpu(type=g)
        elif isinstance(g, dict) and "type" in g:
            gpus[str(alias)] = Gpu(type=str(g["type"]), partition=g.get("partition"))
        else:
            raise ConfigError(f"cluster '{name}': gpu alias '{alias}' must be a type name or {{type, partition}}")

    defaults = {}
    for key, value in (spec.get("defaults") or {}).items():
        key = str(key).lstrip("-")
        if key == "time" and not isinstance(value, str):
            raise ConfigError(f"cluster '{name}': quote the time value in defaults (YAML reads 4:00:00 as a number)")
        defaults[key] = str(value)

    env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
    scratch = str(spec.get("scratch") or "~/mslurm").rstrip("/")

    return Cluster(
        name=name,
        host=str(spec["host"]),
        user=spec.get("user"),
        scratch=scratch,
        account=spec.get("account"),
        qos=spec.get("qos"),
        partition=spec.get("partition"),
        gpus=gpus,
        setup=str(spec.get("setup") or "").strip(),
        defaults=defaults,
        env=env,
    )


def load(path: str = CONFIG_PATH) -> Config:
    if not os.path.exists(path):
        raise ConfigError(f"no config at {path}. Run `mslurm init` to create one from the example.")
    with open(path) as f:
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: {exc}")
    return parse(raw)
