"""The budget has to be right about containers, because that is where it is
wrong in the way that kills a job."""

import math

import pytest

from vtea_core.blocked import (
    CGROUP,
    DETECTED,
    ENV,
    ENV_VAR,
    FALLBACK,
    USER,
    MemoryBudget,
    detect_memory_budget,
    format_bytes,
    parse_size,
)
from vtea_core.blocked import budget as budget_module

GB = 1024**3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2048", 2048),
        ("8G", 8 * 1000**3),
        ("8GiB", 8 * 1024**3),
        ("512MiB", 512 * 1024**2),
        ("1.5 gb", 1_500_000_000),
        ("  4 KiB ", 4096),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "eight gigs", "8 furlongs", "-4G", "0"])
def test_parse_size_rejects_nonsense(text):
    with pytest.raises(ValueError):
        parse_size(text)


def test_format_bytes_is_readable():
    assert format_bytes(4.8 * GB) == "4.8 GiB"
    assert format_bytes(512) == "512 B"
    assert format_bytes(None) == "unknown"


def test_usable_is_the_fraction_divided_between_workers():
    budget = MemoryBudget(10 * GB, fraction=0.5, workers=2)
    assert budget.usable_bytes == int(10 * GB * 0.5 / 2)


def test_workers_do_not_divide_the_gpu():
    # Four CPU workers still share one device, so dividing the GPU budget by
    # them would under-use it fourfold.
    budget = MemoryBudget(10 * GB, fraction=0.5, workers=4, gpu_bytes=8 * GB)
    assert budget.gpu_usable_bytes() == int(8 * GB * 0.5)


def test_no_gpu_has_no_gpu_budget():
    assert MemoryBudget(GB).gpu_usable_bytes() is None
    assert not MemoryBudget(GB).has_gpu


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"total_bytes": 0},
        {"total_bytes": -1},
        {"total_bytes": GB, "fraction": 0},
        {"total_bytes": GB, "fraction": 1.5},
        {"total_bytes": GB, "workers": 0},
    ],
)
def test_impossible_budgets_are_refused(kwargs):
    with pytest.raises(ValueError):
        MemoryBudget(**kwargs)


def test_a_fallback_budget_says_it_is_one(monkeypatch):
    monkeypatch.setattr(budget_module, "cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(budget_module, "available_bytes", lambda: None)
    monkeypatch.delenv(ENV_VAR, raising=False)
    budget = detect_memory_budget()
    assert budget.source == FALLBACK
    assert not budget.is_measured
    assert "fallback" in budget.describe()


def test_the_container_limit_beats_what_the_host_reports(monkeypatch):
    # The failure this exists to prevent: psutil reports the host's 256 GB
    # from inside an 8 GB container, and the plan gets OOM-killed.
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(budget_module, "cgroup_limit_bytes", lambda: 8 * GB)
    monkeypatch.setattr(budget_module, "available_bytes", lambda: 256 * GB)
    budget = detect_memory_budget()
    assert budget.total_bytes == 8 * GB
    assert budget.source == CGROUP


def test_a_generous_container_limit_does_not_override_a_busy_machine(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(budget_module, "cgroup_limit_bytes", lambda: 64 * GB)
    monkeypatch.setattr(budget_module, "available_bytes", lambda: 2 * GB)
    budget = detect_memory_budget()
    assert budget.total_bytes == 2 * GB
    assert budget.source == DETECTED


def test_the_environment_variable_wins_over_detection(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "12GiB")
    monkeypatch.setattr(budget_module, "cgroup_limit_bytes", lambda: 8 * GB)
    budget = detect_memory_budget()
    assert budget.total_bytes == 12 * GB
    assert budget.source == ENV


def test_a_typo_in_the_environment_variable_falls_through(monkeypatch):
    # Better to plan against the machine than to hand a run the whole host
    # because someone wrote "8 gigs", and better than aborting a batch job.
    monkeypatch.setenv(ENV_VAR, "8 gigs")
    monkeypatch.setattr(budget_module, "cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(budget_module, "available_bytes", lambda: 16 * GB)
    assert detect_memory_budget().source == DETECTED


def test_an_explicit_budget_wins_over_everything(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "12GiB")
    assert detect_memory_budget(total_bytes=3 * GB).source == USER


def test_cgroup_v2_unlimited_reads_as_no_limit(tmp_path, monkeypatch):
    path = tmp_path / "memory.max"
    path.write_text("max\n")
    monkeypatch.setattr(budget_module, "_CGROUP_V2_PATHS", (str(path),))
    monkeypatch.setattr(budget_module, "_CGROUP_V1_PATHS", ())
    assert budget_module.cgroup_limit_bytes() is None


def test_cgroup_v2_limit_is_read(tmp_path, monkeypatch):
    path = tmp_path / "memory.max"
    path.write_text(f"{8 * GB}\n")
    monkeypatch.setattr(budget_module, "_CGROUP_V2_PATHS", (str(path),))
    assert budget_module.cgroup_limit_bytes() == 8 * GB


def test_cgroup_v1_sentinel_is_not_a_limit(tmp_path, monkeypatch):
    # v1 spells "unlimited" as a number near 2**63 rather than as a word.
    path = tmp_path / "memory.limit_in_bytes"
    path.write_text("9223372036854771712\n")
    monkeypatch.setattr(budget_module, "_CGROUP_V2_PATHS", ())
    monkeypatch.setattr(budget_module, "_CGROUP_V1_PATHS", (str(path),))
    assert budget_module.cgroup_limit_bytes() is None


def test_a_missing_cgroup_file_is_not_an_error(monkeypatch):
    monkeypatch.setattr(budget_module, "_CGROUP_V2_PATHS", ("/nonexistent/memory.max",))
    monkeypatch.setattr(budget_module, "_CGROUP_V1_PATHS", ())
    assert budget_module.cgroup_limit_bytes() is None


def test_available_bytes_is_plausible_on_this_machine():
    value = budget_module.available_bytes()
    assert value is None or value > 0


def test_gpu_probe_survives_having_no_gpu():
    value = budget_module.gpu_free_bytes()
    assert value is None or value > 0


def test_describe_mentions_the_source_and_the_numbers():
    text = MemoryBudget(8 * GB, source=USER).describe()
    assert "8.0 GiB" in text
    assert "set by you" in text


def test_describe_reports_workers_when_there_are_several():
    assert "4 workers" in MemoryBudget(8 * GB, workers=4).describe()
    assert "workers" not in MemoryBudget(8 * GB).describe()


def test_the_fallback_matches_the_placeholder_it_replaces():
    # data.volume.DEFAULT_MEMORY_BUDGET_BYTES was the hardcoded stand-in;
    # an undetectable machine should behave as it always has, not newly
    # differently.
    from vtea_core.data.volume import DEFAULT_MEMORY_BUDGET_BYTES

    assert budget_module.FALLBACK_TOTAL_BYTES == DEFAULT_MEMORY_BUDGET_BYTES


def test_with_workers_and_with_total_do_not_mutate():
    budget = MemoryBudget(8 * GB)
    assert budget.with_workers(4).workers == 4
    assert budget.workers == 1
    assert budget.with_total(GB).total_bytes == GB
    assert budget.total_bytes == 8 * GB


def test_format_bytes_caps_at_tebibytes():
    assert format_bytes(4 * 1024**5).endswith("TiB")
    assert not math.isnan(float(format_bytes(1024**4).split()[0]))
