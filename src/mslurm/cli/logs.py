"""mslurm logs: tail a job's output file over ssh."""

from __future__ import annotations

import shlex

from mslurm.cli.common import find_workdir, load_config, resolve_refs
from mslurm.ids import JobRef, output_name
from mslurm.ssh import Remote, sh_path


def add_parser(sub):
    p = sub.add_parser("logs", aliases=["log"], help="show or follow a job's output")
    p.add_argument("job", metavar="JOB")
    p.add_argument("-M", "--cluster", help="cluster for a bare job id")
    p.add_argument("-f", "--follow", action="store_true", help="keep following (waits for the file to appear)")
    p.add_argument("-n", "--lines", type=int, default=100)
    p.add_argument("--err", action="store_true", help="the stderr file instead of stdout")
    p.set_defaults(run=run)


def tail(remote: Remote, ref: JobRef, workdir: str, *, follow: bool = False, lines: int = 100, stderr: bool = False) -> int:
    key = "StdErr" if stderr else "StdOut"
    fallback = f"{sh_path(workdir)}/{output_name(ref.job_id, stderr)}"
    jid = shlex.quote(ref.job_id)
    script = f"""
f=$(scontrol show job {jid} 2>/dev/null | sed -n 's/^ *{key}=//p' | head -1)
[ -n "$f" ] || f={fallback}
if [ ! -e "$f" ]; then
  if {'true' if follow else 'false'}; then
    echo "waiting for $f" >&2
    while [ ! -e "$f" ]; do sleep 3; done
  else
    echo "no output file yet: $f" >&2; exit 3
  fi
fi
exec tail -n {int(lines)} {'-F' if follow else ''} -- "$f"
"""
    return remote.stream(script)


def run(args) -> int:
    cfg = load_config()
    (ref,) = resolve_refs(cfg, [args.job], args.cluster)
    remote = Remote(cfg.get(ref.cluster))
    workdir = find_workdir(remote, ref)
    return tail(remote, ref, workdir, follow=args.follow, lines=args.lines, stderr=args.err)
