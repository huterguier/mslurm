"""Talking to a cluster's login node: ssh with a reused connection, plus rsync over it.

Every cluster gets one ControlMaster socket under ~/.cache/mslurm/ssh. The first
command opens the connection and later ones ride on it, so a command like
`mslurm queue` costs one round trip per cluster, not one handshake per call.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

from mslurm import MslurmError
from mslurm.config import CACHE_DIR, Cluster

CONNECT_TIMEOUT = 10
CONTROL_PERSIST = "1h"


class RemoteError(MslurmError):
    def __init__(self, cluster: Cluster, returncode: int, detail: str):
        self.cluster = cluster
        self.returncode = returncode
        msg = f"{cluster.name}: ssh {cluster.target} exited {returncode}: {detail or 'no output'}"
        if returncode == 255:
            msg += (
                f"\n  (could not connect. If {cluster.host} needs a password or 2FA, run "
                f"`mslurm ssh {cluster.name}` once; later commands reuse that connection.)"
            )
        super().__init__(msg)


def sh_path(path: str) -> str:
    """Quote a remote path for the shell, letting a leading ~ or $VAR expand."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"' + shlex.quote(path[1:])
    if path.startswith("$"):
        head, sep, rest = path.partition("/")
        return f'"{head}"' + (shlex.quote(sep + rest) if rest else "")
    return shlex.quote(path)


class Remote:
    def __init__(self, cluster: Cluster):
        self.cluster = cluster

    @property
    def socket(self) -> str:
        return os.path.join(CACHE_DIR, "ssh", self.cluster.name)

    def ssh_options(self, batch: bool = True) -> list[str]:
        os.makedirs(os.path.dirname(self.socket), mode=0o700, exist_ok=True)
        return [
            *(["-o", "BatchMode=yes"] if batch else []),
            "-o", f"ConnectTimeout={CONNECT_TIMEOUT}",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.socket}",
            "-o", f"ControlPersist={CONTROL_PERSIST}",
            "-o", "ServerAliveInterval=30",
        ]

    def _command(self, script: str, tty: bool = False) -> list[str]:
        # Wrapping in `bash -c` keeps our scripts working when the login shell is zsh or fish.
        # csh/tcsh login shells are not supported: they reject the newlines inside the quotes.
        args = ["ssh", *self.ssh_options()]
        if tty:
            args.append("-t")
        args += [self.cluster.target, "bash -c " + shlex.quote(script)]
        return args

    def run(
        self,
        script: str,
        *,
        input: str | bytes | None = None,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a bash script on the login node and capture its output."""
        binary = isinstance(input, bytes)
        proc = subprocess.run(
            self._command(script),
            input=input if input is not None else ("" if not binary else b""),
            capture_output=True,
            text=not binary,
            timeout=timeout,
        )
        if binary:
            proc = subprocess.CompletedProcess(
                proc.args, proc.returncode, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")
            )
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RemoteError(self.cluster, proc.returncode, detail)
        return proc

    def stream(self, script: str, tty: bool = False) -> int:
        """Run a script with our stdin/stdout attached, for tails and interactive things."""
        try:
            return subprocess.call(self._command(script, tty=tty))
        except KeyboardInterrupt:
            return 130

    def exec_shell(self, command: list[str] | None = None) -> None:
        """Replace this process with an interactive ssh session (or a remote command)."""
        args = ["ssh", *self.ssh_options(batch=False), self.cluster.target, *(command or [])]
        sys.stdout.flush()
        os.execvp("ssh", args)

    def rsync_from(self, remote_dir: str, local_dir: str, excludes: list[str] = (), extra: list[str] = ()) -> None:
        """Copy a remote directory into local_dir. Local files are never deleted."""
        os.makedirs(local_dir, exist_ok=True)
        args = [
            "rsync", "-az", "--human-readable", "--info=name1,stats1",
            *(f"--exclude={pattern}" for pattern in excludes),
            *extra,
            "-e", shlex.join(["ssh", *self.ssh_options()]),
            f"{self.cluster.target}:{remote_dir.rstrip('/')}/",
            f"{local_dir.rstrip('/')}/",
        ]
        proc = subprocess.run(args)
        if proc.returncode != 0:
            raise RemoteError(self.cluster, proc.returncode, "rsync failed")
