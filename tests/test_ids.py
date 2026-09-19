import pytest

from mslurm import MslurmError
from mslurm.ids import JobRef, output_name, parse_ref


def test_parse_ref():
    assert parse_ref("alpha:123") == ("alpha", "123")
    assert parse_ref("123") == (None, "123")
    assert parse_ref("alpha:123_4") == ("alpha", "123_4")
    assert parse_ref("alpha:123_[0-9]") == ("alpha", "123_[0-9]")
    assert parse_ref("alpha:123_[0-9%2]") == ("alpha", "123_[0-9%2]")
    assert parse_ref("alpha:123+1") == ("alpha", "123+1")


@pytest.mark.parametrize("bad", ["", "abc", "alpha:", "alpha:12a", "12_", "alpha:123:4"])
def test_parse_ref_rejects(bad):
    with pytest.raises(MslurmError):
        parse_ref(bad)


def test_jobref():
    ref = JobRef("alpha", "123_4")
    assert str(ref) == "alpha:123_4"
    assert ref.base == "123"
    assert ref.is_array_task
    assert not JobRef("alpha", "123_[0-3]").is_array_task
    assert JobRef("alpha", "123+1").base == "123"


def test_output_name():
    assert output_name("123") == "slurm-123.out"
    assert output_name("123_4") == "slurm-123_4.out"
    assert output_name("123", stderr=True) == "slurm-123.err"
