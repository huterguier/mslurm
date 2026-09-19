"""mslurm: one CLI for several Slurm clusters, talking to each over SSH."""

from importlib.metadata import version

__version__ = version("mslurm")


class MslurmError(Exception):
    """Any error the CLI should report as a one-line message instead of a traceback."""
