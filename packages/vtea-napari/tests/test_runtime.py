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
    def test_install_torch_flag_without_a_gpu_installs_cpu(self):
        """Pinned rather than relying on the test machine having no GPU -
        otherwise this passes for the wrong reason on CI and fails on a
        workstation."""
        from vtea_napari.app import main

        with patch("vtea_napari.runtime.detect_torch_variant", return_value="cpu"):
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


class TestCudaDetection:
    def _nvidia_smi_output(self, cuda_version="12.4"):
        return (
            "Fri Aug 25 02:00:00 2026\n"
            "+---------------------------------------------------------------+\n"
            f"| NVIDIA-SMI 550.54.14  Driver Version: 550.54.14  CUDA Version: {cuda_version} |\n"
        )

    def test_reads_the_driver_cuda_version(self):
        from vtea_napari.runtime import detect_driver_cuda

        with patch("vtea_napari.runtime.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("vtea_napari.runtime.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = self._nvidia_smi_output("12.4")
                assert detect_driver_cuda() == (12, 4)

    def test_no_nvidia_smi_means_no_gpu(self):
        from vtea_napari.runtime import detect_driver_cuda

        with patch("vtea_napari.runtime.shutil.which", return_value=None):
            assert detect_driver_cuda() is None

    def test_nvidia_smi_failure_is_not_fatal(self):
        from vtea_napari.runtime import detect_driver_cuda

        with patch("vtea_napari.runtime.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("vtea_napari.runtime.subprocess.run", side_effect=OSError("boom")):
                assert detect_driver_cuda() is None

    def test_picks_the_newest_wheel_the_driver_supports(self):
        from vtea_napari.runtime import detect_torch_variant

        # A 12.4 driver must not get a cu126/cu128 wheel.
        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=(12, 4)):
            assert detect_torch_variant() == "cu124"
        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=(12, 2)):
            assert detect_torch_variant() == "cu121"
        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=(13, 0)):
            assert detect_torch_variant() == "cu128"

    def test_falls_back_to_cpu(self):
        from vtea_napari.runtime import detect_torch_variant

        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=None):
            assert detect_torch_variant() == "cpu"
        # A driver older than every published CUDA wheel.
        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=(10, 2)):
            assert detect_torch_variant() == "cpu"

    def test_install_torch_without_a_variant_auto_detects(self):
        from vtea_napari.app import main

        with patch("vtea_napari.runtime.detect_torch_variant", return_value="cu124"):
            with patch("vtea_napari.runtime.install_torch", return_value=0) as install:
                assert main(["--install-torch"]) == 0
        install.assert_called_once_with("cu124")


class TestGpuStatus:
    def test_reports_missing_torch_with_the_next_command(self, tmp_path, monkeypatch, capsys):
        from vtea_napari.runtime import gpu_status

        monkeypatch.setenv(TORCH_PATH_ENV, str(tmp_path / "absent"))
        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=(12, 4)):
            with patch.dict(sys.modules, {"torch": None}):
                assert gpu_status() == 0
        out = capsys.readouterr().out
        assert "CUDA 12.4" in out
        assert "--install-torch" in out

    def test_reports_a_working_gpu(self, tmp_path, monkeypatch, capsys):
        from unittest.mock import MagicMock

        from vtea_napari.runtime import gpu_status

        monkeypatch.setenv(TORCH_PATH_ENV, str(tmp_path))
        fake_torch = MagicMock()
        fake_torch.__version__ = "2.6.0+cu124"
        fake_torch.__file__ = str(tmp_path / "torch" / "__init__.py")
        fake_torch.version.cuda = "12.4"
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_name.return_value = "NVIDIA RTX A4000"

        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=(12, 4)):
            with patch("vtea_napari.runtime.torch_is_bundled", return_value=False):
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    assert gpu_status() == 0
        out = capsys.readouterr().out
        assert "RTX A4000" in out
        assert "run on the GPU" in out

    def test_tells_deeplearning_users_to_switch_builds(self, capsys):
        from unittest.mock import MagicMock

        from vtea_napari.runtime import gpu_status

        fake_torch = MagicMock()
        fake_torch.__version__ = "2.13.0+cpu"
        fake_torch.version.cuda = None

        with patch("vtea_napari.runtime.detect_driver_cuda", return_value=(12, 4)):
            with patch("vtea_napari.runtime.torch_is_bundled", return_value=True):
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    assert gpu_status() == 0
        out = capsys.readouterr().out
        assert "CPU-only torch baked in" in out
        assert "slim download" in out
