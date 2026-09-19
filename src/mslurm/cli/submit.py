"""mslurm submit: snapshot the git tree, ship it, and sbatch a job in a fresh job directory."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import socket
import time

from mslurm import MslurmError, index, ship
from mslurm.cli import logs
from mslurm.cli.common import load_config, warn
from mslurm.ids import JobRef
from mslurm.render import Resources, guess_name, render_script, sbatch_args, sbatch_directives, sbatch_options, slug
from mslurm.ssh import Remote, sh_path


def add_parser(sub):
    p = sub.add_parser(
        "submit",
        help="ship the current git tree and submit a job",
        usage="mslurm submit [options] -- COMMAND [ARG...]\n       mslurm submit [options] SCRIPT.sh [-- SCRIPT-ARG...]",
        description="Snapshots the git repository you are in (uncommitted changes included), uploads it once per "
        "content hash, copies it into a new job directory on the cluster and runs sbatch there. Without a "
        "SCRIPT the sbatch script is generated from the options below and the cluster profile. "
        "Short flags mean what they mean for sbatch.",
    )
    p.add_argument("script", nargs="?", help="an existing sbatch script to submit as-is (its #SBATCH lines win)")
    p.add_argument("-M", "--cluster", help="cluster profile from clusters.yaml (default: `default:` there)")
    p.add_argument("-J", "--name", "--job-name", dest="name", help="job name (default: guessed from the command)")
    p.add_argument("-t", "--time", help="wall time: 30m, 2h, 1d or Slurm's HH:MM:SS / D-HH:MM:SS")
    p.add_argument("-G", "--gpus", help="GPUs: N, TYPE or TYPE:N (TYPE may be an alias from clusters.yaml)")
    p.add_argument("-p", "--partition")
    p.add_argument("-c", "--cpus", "--cpus-per-task", dest="cpus", help="cpus per task")
    p.add_argument("--mem", help="memory per node, e.g. 32G")
    p.add_argument("-N", "--nodes")
    p.add_argument("-n", "--ntasks")
    p.add_argument("-a", "--array", help="array spec, e.g. 0-9 or 0-99%%10")
    p.add_argument("-A", "--account")
    p.add_argument("-q", "--qos")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment variable for the job")
    p.add_argument("--sbatch", action="append", default=[], metavar="FLAG", help="extra sbatch flag, e.g. --sbatch=--exclusive")
    p.add_argument("-f", "--follow", action="store_true", help="tail the job output after submitting")
    p.add_argument("--dry-run", action="store_true", help="print the generated script and stop")
    p.set_defaults(run=run)


def run(args) -> int:
    cfg = load_config()
    cluster = cfg.pick(args.cluster)
    remote = Remote(cluster)

    script_path = args.script
    if script_path is not None and not os.path.isfile(script_path):
        raise MslurmError(f"'{script_path}' is not a file. Put a command after `--`, e.g. mslurm submit -- python train.py")
    command = [] if script_path else args.rest
    if not script_path and not command:
        raise MslurmError("nothing to run: give a script or put a command after `--`")

    env = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep:
            raise MslurmError(f"--env expects KEY=VALUE, got '{item}'")
        env[key] = value

    snap = ship.snapshot()
    fallback = os.path.basename(snap.root)
    if args.name:
        name = args.name
    elif script_path:
        name = slug(os.path.splitext(os.path.basename(script_path))[0])
    else:
        name = guess_name(command, fallback)

    res = Resources(
        name=name if not script_path or args.name else None,
        time=args.time, gpus=args.gpus, partition=args.partition, cpus=args.cpus, mem=args.mem,
        nodes=args.nodes, ntasks=args.ntasks, array=args.array, account=args.account, qos=args.qos,
        extra=args.sbatch, env=env,
    )
    comment = f"mslurm:{snap.short}"
    code = ship.code_dir(cluster, snap.tree)

    if script_path:
        with open(script_path) as f:
            script_text = f.read()
        script_has = sbatch_directives(script_text)
        flags = sbatch_args(sbatch_options(cluster, res, comment, script_mode=True, script_has=script_has))
        if env:
            flags.append("--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items()))
        script_args = args.rest
        shown_command = shlex.join([script_path, *script_args])
    else:
        script_text = render_script(cluster, res, command, code=code, comment=comment)
        flags, script_args = [], []
        shown_command = shlex.join(command)

    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(name, 30)}-{secrets.token_hex(2)}"
    job_dir = cluster.path("jobs", stamp)
    sbatch_cmd = ["sbatch", "--parsable", *flags, "mslurm.sh", *script_args]

    if args.dry_run:
        print(script_text, end="")
        print(f"# would run in {cluster.name}:{job_dir}/{snap.subdir}".rstrip("/"))
        print(f"# {shlex.join(sbatch_cmd)}")
        return 0

    if snap.dirty:
        warn(f"working tree has uncommitted changes; shipping them as snapshot {snap.short}")
    ship.ensure_uploaded(remote, snap, log=lambda m: print(m, flush=True))

    meta = {
        "cluster": cluster.name,
        "name": name,
        "command": shown_command,
        "code": snap.tree,
        "head": snap.head,
        "dirty": snap.dirty,
        "subdir": snap.subdir,
        "local_root": snap.root,
        "local_host": socket.gethostname(),
        "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    job_id, workdir = _submit_remote(remote, code, job_dir, snap.subdir, script_text, meta, sbatch_cmd)

    ref = JobRef(cluster.name, job_id)
    index.record(index.JobRecord(
        cluster=cluster.name, job_id=job_id, name=name, command=shown_command, workdir=workdir,
        code=snap.tree, head=snap.head, dirty=snap.dirty, local_root=snap.root, subdir=snap.subdir,
    ))
    summary = "  ".join(f"{k}={v}" for k, v in sbatch_options(cluster, res, script_mode=bool(script_path))
                        if k in ("partition", "gres", "time", "array") and v)
    print(f"Submitted {ref}  {name}  {summary}".rstrip())
    print(f"  logs: mslurm logs {ref}    dir: {workdir}")
    if args.follow:
        return logs.tail(remote, ref, workdir, follow=True)
    return 0


def _submit_remote(remote, code, job_dir, subdir, script_text, meta, sbatch_cmd) -> tuple[str, str]:
    """Create the job dir from the code snapshot, write the script, sbatch it. Returns (job id, workdir)."""
    body = json.dumps(meta, indent=2)
    if "MSLURM_JSON" in body:
        raise MslurmError("metadata contains the heredoc delimiter; rename your job")
    work = '"$dir"' + (f"/{shlex.quote(subdir)}" if subdir else "")
    script = f"""
set -e
code={sh_path(code)}
dir={sh_path(job_dir)}
mkdir -p "$(dirname "$dir")"
mkdir "$dir"
cp -a "$code"/. "$dir"/
cat > "$dir/mslurm.json" <<'MSLURM_JSON'
{body}
MSLURM_JSON
cd {work}
cat > mslurm.sh
chmod +x mslurm.sh
if ! id=$({shlex.join(sbatch_cmd)}); then rm -rf "$dir"; exit 1; fi
echo "$id" > "$dir/mslurm.jobid"
echo "$id"
pwd -P
"""
    out = remote.run(script, input=script_text).stdout.strip().splitlines()
    if len(out) < 2:
        raise MslurmError(f"unexpected reply from {remote.cluster.name}: {out}")
    job_id = out[-2].split(";")[0].strip()
    return job_id, out[-1].strip()
