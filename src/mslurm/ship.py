"""Snapshot the local git working tree and make sure the cluster has a copy.

The snapshot is a git tree object built from the working tree (tracked files,
plus untracked ones that are not ignored), so uncommitted edits are included.
Its hash names the copy on the cluster, `<scratch>/code/<tree>`, which means a
second submit of the same code uploads nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from mslurm import MslurmError
from mslurm.config import Cluster
from mslurm.ssh import Remote, sh_path


class ShipError(MslurmError):
    pass


@dataclass(frozen=True)
class Snapshot:
    root: str  # local repository root
    subdir: str  # cwd relative to root, "" at the root
    tree: str  # tree hash of the working tree
    head: str | None  # HEAD commit, None in a repo without commits
    dirty: bool  # tree differs from HEAD's tree

    @property
    def short(self) -> str:
        return self.tree[:10]


def _git(args: list[str], cwd: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise ShipError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def snapshot(cwd: str | None = None) -> Snapshot:
    cwd = os.path.abspath(cwd or os.getcwd())
    top = _git(["rev-parse", "--show-toplevel"], cwd, check=False)
    if top.returncode != 0:
        raise ShipError(f"{cwd} is not inside a git repository (mslurm ships code as a git snapshot; run `git init`)")
    root = top.stdout.strip()
    subdir = os.path.relpath(cwd, root)
    subdir = "" if subdir == "." else subdir

    with tempfile.TemporaryDirectory(prefix="mslurm-") as tmp:
        index = os.path.join(tmp, "index")
        current = _git(["rev-parse", "--git-path", "index"], root).stdout.strip()
        current = current if os.path.isabs(current) else os.path.join(root, current)
        if os.path.exists(current):
            shutil.copy(current, index)  # keeps the stat cache, so `add -A` is fast on big trees
        env = {**os.environ, "GIT_INDEX_FILE": index}
        _git(["add", "-A", "--", "."], root, env=env)
        tree = _git(["write-tree"], root, env=env).stdout.strip()

    head_proc = _git(["rev-parse", "--verify", "-q", "HEAD"], root, check=False)
    head = head_proc.stdout.strip() or None
    head_tree = _git(["rev-parse", "HEAD^{tree}"], root, check=False).stdout.strip() if head else None
    return Snapshot(root=root, subdir=subdir, tree=tree, head=head, dirty=tree != head_tree)


def archive(snap: Snapshot) -> bytes:
    proc = subprocess.run(["git", "archive", "--format=tar.gz", snap.tree], cwd=snap.root, capture_output=True)
    if proc.returncode != 0:
        raise ShipError(f"git archive failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def code_dir(cluster: Cluster, tree: str) -> str:
    return cluster.path("code", tree)


def ensure_uploaded(remote: Remote, snap: Snapshot, log=print) -> bool:
    """Upload the snapshot unless the cluster already has it. Returns True if it uploaded."""
    cluster = remote.cluster
    target = sh_path(code_dir(cluster, snap.tree))
    if remote.run(f"test -d {target}", check=False).returncode == 0:
        return False

    data = archive(snap)
    log(f"uploading code {snap.short} ({_human(len(data))}) to {cluster.name}")
    script = f"""
set -e
code={sh_path(cluster.path("code"))}
mkdir -p "$code"
tmp="$code/.tmp-{snap.tree}-$$"
mkdir -p "$tmp"
tar -xzf - -C "$tmp"
if ! mv -T "$tmp" "$code/{snap.tree}" 2>/dev/null; then
  rm -rf "$tmp"
  test -d "$code/{snap.tree}"
fi
"""
    remote.run(script, input=data)
    return True


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
