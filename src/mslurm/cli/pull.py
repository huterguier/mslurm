"""mslurm pull: copy a job's outputs to ./runs/<cluster>/<jobid>/, never into the source tree."""

from __future__ import annotations

import os
import subprocess
import tempfile

from mslurm import index
from mslurm.cli.common import find_workdir, load_config, read_meta, resolve_refs, warn
from mslurm.ship import code_dir
from mslurm.ssh import Remote, sh_path

ALWAYS_EXCLUDE = [".venv/", "__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/"]


def add_parser(sub):
    p = sub.add_parser("pull", help="fetch a job's outputs (files the job created or changed)")
    p.add_argument("job", metavar="JOB")
    p.add_argument("dest", nargs="?", help="destination directory (default runs/<cluster>/<jobid>)")
    p.add_argument("-M", "--cluster", help="cluster for a bare job id")
    p.add_argument("--all", action="store_true", help="also copy the code snapshot the job ran from")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(run=run)


def run(args) -> int:
    cfg = load_config()
    (ref,) = resolve_refs(cfg, [args.job], args.cluster)
    cluster = cfg.get(ref.cluster)
    remote = Remote(cluster)

    rec = index.get(ref.cluster, ref.job_id)
    workdir = rec.workdir if rec else find_workdir(remote, ref)
    dest = args.dest or os.path.join(_repo_root(), "runs", ref.cluster, ref.job_id)

    excludes = list(ALWAYS_EXCLUDE)
    if not args.all:
        meta = read_meta(remote, workdir)
        code = (rec and rec.code) or meta.get("code")
        subdir = (rec.subdir if rec else meta.get("subdir")) or ""
        if code:
            snapshot_dir = sh_path(code_dir(cluster, code)) + (f"/{subdir}" if subdir else "")
            listing = remote.run(f"cd {snapshot_dir} 2>/dev/null && find . -type f", check=False).stdout
            excludes += ["/" + line[2:] for line in listing.splitlines() if line.startswith("./")]
        else:
            warn("no code snapshot recorded for this job; pulling the whole directory")

    extra = ["--dry-run"] if args.dry_run else []
    dest = os.path.relpath(dest) if not os.path.isabs(args.dest or "") else dest
    with tempfile.NamedTemporaryFile("w", prefix="mslurm-exclude-", suffix=".txt", delete=False) as f:
        f.write("\n".join(excludes) + "\n")
        exclude_file = f.name
    try:
        print(f"{ref}:{workdir} -> {dest}")
        remote.rsync_from(workdir, dest, extra=["--prune-empty-dirs", f"--exclude-from={exclude_file}", *extra])
    finally:
        os.unlink(exclude_file)
    return 0


def _repo_root() -> str:
    """The git top level, so pulls land in one place however deep you are; else the cwd."""
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else os.getcwd()
