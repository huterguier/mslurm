"""Entry point: `mslurm COMMAND ...`."""

from __future__ import annotations

import argparse
import sys

from mslurm import MslurmError, __version__
from mslurm.cli import cancel, clean, info, init, logs, pull, queue, shell, show, submit

COMMANDS = (submit, queue, logs, show, cancel, pull, info, clean, shell, init)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mslurm",
        description="Submit to and inspect several Slurm clusters over SSH.",
        epilog="Anything after `--` is passed through: the job command for `mslurm submit`, a remote command for `mslurm ssh`.",
    )
    parser.add_argument("--version", action="version", version=f"mslurm {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for module in COMMANDS:
        module.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1 :]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.rest = rest
    try:
        return int(args.run(args) or 0)
    except MslurmError as exc:
        print(f"mslurm: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
