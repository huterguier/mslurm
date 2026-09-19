import pytest
import yaml

from mslurm.config import EXAMPLE_PATH, ConfigError, parse


def test_example_parses():
    with open(EXAMPLE_PATH) as f:
        cfg = parse(yaml.safe_load(f))
    alpha = cfg.get("alpha")
    assert cfg.default == "alpha"
    assert alpha.host == "alpha"
    assert alpha.account == "my-account"
    assert alpha.defaults["time"] == "4:00:00"
    assert alpha.defaults["cpus-per-task"] == "4"
    assert alpha.gpus["3090"].type == "rtx3090"
    assert alpha.gpus["small"].partition == "interactive"
    assert "uv sync" in alpha.setup
    assert alpha.path("code", "abc") == "~/mslurm/code/abc"


def test_pick_single_cluster_is_default(monkeypatch):
    monkeypatch.delenv("MSLURM_CLUSTER", raising=False)
    cfg = parse({"clusters": {"a": {"host": "a"}}})
    assert cfg.pick(None).name == "a"


def test_pick_needs_name_with_several(monkeypatch):
    monkeypatch.delenv("MSLURM_CLUSTER", raising=False)
    cfg = parse({"clusters": {"a": {"host": "a"}, "b": {"host": "b"}}})
    with pytest.raises(ConfigError):
        cfg.pick(None)
    monkeypatch.setenv("MSLURM_CLUSTER", "b")
    assert cfg.pick(None).name == "b"
    assert cfg.pick("a").name == "a"


@pytest.mark.parametrize(
    "raw",
    [
        {"clusters": {"a": {}}},  # no host
        {"clusters": {"a": {"host": "a", "ssh_host": "x"}}},  # unknown key
        {"clusters": {"a": {"host": "a", "defaults": {"time": 14400}}}},  # unquoted time
        {"clusters": {"a": {"host": "a"}}, "default": "b"},
        {"clusters": {"a": {"host": "a", "gpus": {"x": {"partition": "p"}}}}},
    ],
)
def test_rejects_bad_config(raw):
    with pytest.raises(ConfigError):
        parse(raw)
