from mslurm.__main__ import build_parser
from mslurm.cli.clean import _script

COMMANDS = ("submit", "queue", "logs", "show", "cancel", "pull", "info", "clean", "ssh", "init")


def test_every_command_is_registered():
    subparsers = next(a for a in build_parser()._actions if a.dest == "command")
    assert set(COMMANDS) <= set(subparsers.choices)


def test_clean_script_only_deletes_when_asked():
    class Cluster:
        def path(self, *parts):
            return "~/mslurm/" + "/".join(parts)

    dry = _script(Cluster(), 14, delete=False)
    wet = _script(Cluster(), 14, delete=True)
    assert "rm -rf" not in dry
    assert 'rm -rf -- "$d"' in wet
    assert "-name '.tmp-*'" in wet
