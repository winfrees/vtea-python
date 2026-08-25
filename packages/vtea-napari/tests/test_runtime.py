import sys
from pathlib import Path
from unittest.mock import patch

from vtea_napari.runtime import (
    TORCH_PATH_ENV,
    activate_external_torch,
    default_runtime_dir,
    install_torch,
    torch_runtime_dir,
)


class TestRuntimeDir:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TORCH_PATH_ENV, str(tmp_path))
        assert torch_runtime_dir() == tmp_path

    def test_falls_back_to_a_user_directory(self, monkeypatch):
        monkeypatch.delenv(TORCH_PATH_ENV, raising=False)
        assert torch_runtime_dir() == default_runtime_dir()

    def test_user_directory_is_under_the_home_dir(self, monkeypatch):
        monkeypatch.delenv(TORCH_PATH_ENV, raising=False)
        assert "vtea" in default_runtime_dir().parts


class TestActivateExternalTorch:
    def test_missing_directory_is_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TORCH_PATH_ENV, str(tmp_path / "nope"))
        before = list(sys.path)
        assert activate_external_torch() is None
        assert sys.path == before

    def test_existing_directory_goes_to_the_front_of_sys_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TORCH_PATH_ENV, str(tmp_path))
        monkeypatch.setattr(sys, "path", list(sys.path))
        with patch("vtea_napari.runtime.torch_is_bundled", return_value=False):
            assert activate_external_torch() == tmp_path
        assert sys.path[0] == str(tmp_path.resolve())

    def test_not_added_twice(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TORCH_PATH_ENV, str(tmp_path))
        monkeypatch.setattr(sys, "path", list(sys.path))
        with patch("vtea_napari.runtime.torch_is_bundled", return_value=False):
            activate_external_torch()
            activate_external_torch()
        assert sys.path.count(str(tmp_path.resolve())) == 1

    def test_skipped_when_torch_is_bundled(self, tmp_path, monkeypatch):
        """A bundled torch always wins over sys.path in a frozen app, so
        activating an external one would be misleading rather than useful."""
        monkeypatch.setenv(TORCH_PATH_ENV, str(tmp_path))
        with patch("vtea_napari.runtime.torch_is_bundled", return_value=True):
            assert activate_external_torch() is None


class TestInstallTorch:
    def test_rejects_an_unknown_variant(self, capsys):
        assert install_torch("gpu-please") == 2
        assert "Unknown torch variant" in capsys.readouterr().out

    def test_accepts_cpu_and_cuda_variants(self, tmp_path):
        for variant in ("cpu", "cu121", "cu124"):
            with patch("vtea_napari.runtime.subprocess.run") as run:
                run.return_value.returncode = 0
                assert install_torch(variant, target=tmp_path) == 0
            command = run.call_args[0][0]
            assert f"https://download.pytorch.org/whl/{variant}" in command
            assert "torch" in command and "cellpose" in command

    def test_reports_a_missing_python_interpreter(self, tmp_path, capsys):
        with patch("vtea_napari.runtime._host_python", return_value=None):
            assert install_torch("cpu", target=tmp_path) == 2
        assert TORCH_PATH_ENV in capsys.readouterr().out

    def test_propagates_pip_failure(self, tmp_path):
        with patch("vtea_napari.runtime.subprocess.run") as run:
            run.return_value.returncode = 1
            assert install_torch("cpu", target=tmp_path) == 1


class TestCliDispatch:
    def test_install_torch_flag_defaults_to_cpu(self):
        from vtea_napari.app import main

        with patch("vtea_napari.runtime.install_torch", return_value=0) as install:
            assert main(["--install-torch"]) == 0
        install.assert_called_once_with("cpu")

    def test_install_torch_flag_takes_a_variant(self):
        from vtea_napari.app import main

        with patch("vtea_napari.runtime.install_torch", return_value=0) as install:
            assert main(["--install-torch", "cu121"]) == 0
        install.assert_called_once_with("cu121")

    def test_external_torch_is_activated_before_launching(self):
        from unittest.mock import MagicMock

        from vtea_napari.app import main

        with patch("vtea_napari.runtime.activate_external_torch") as activate:
            with patch.dict("sys.modules", {"napari": MagicMock()}):
                assert main([]) == 0
        activate.assert_called_once()


def test_runtime_dir_expands_a_user_path(monkeypatch):
    monkeypatch.setenv(TORCH_PATH_ENV, "~/somewhere")
    assert torch_runtime_dir() == Path.home() / "somewhere"
