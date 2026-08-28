"""Resolving PyTorch (and Cellpose) from outside the bundle.

The standalone download ships without PyTorch. That keeps it small, but the
bigger reason is that a *bundled* torch can never be swapped: PyInstaller's
frozen importer sits ahead of the normal path finder, so `import torch`
inside a frozen app always resolves to the bundled copy. Putting another
torch on PYTHONPATH does nothing - verified directly against a build that
bundled one. Since only one torch build can win, bundling the CPU build
would permanently rule out GPU acceleration.

So torch is left out and resolved at runtime from a directory the user
owns, which supports both ways of getting one:

  * an install VTEA manages, created by `vtea-napari --install-torch`
    (pick `cpu`, or a CUDA build such as `cu121` to use your GPU);
  * an install you already have - a conda env, a venv, a system install -
    by pointing VTEA_TORCH_PATH at its site-packages.

Neither downloads CUDA libraries into the app; the CUDA build of torch
ships its own, and it is installed as an ordinary wheel.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Env var pointing at a site-packages directory that already contains torch.
TORCH_PATH_ENV = "VTEA_TORCH_PATH"

_PYTORCH_INDEX = "https://download.pytorch.org/whl"


def default_runtime_dir() -> Path:
    """Where `--install-torch` puts things when VTEA_TORCH_PATH isn't set."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "vtea" / "torch"


def torch_runtime_dir() -> Path:
    override = os.environ.get(TORCH_PATH_ENV)
    return Path(override).expanduser() if override else default_runtime_dir()


def _bundle_dir() -> Path | None:
    """The directory a frozen build unpacks itself into, or None when not
    running frozen."""
    if not getattr(sys, "frozen", False):
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else None


def _is_inside_bundle(path: str | None) -> bool:
    bundle = _bundle_dir()
    if bundle is None or not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(bundle.resolve())
    except (OSError, ValueError):
        return False


def torch_is_bundled() -> bool:
    """True when torch is frozen into *this build*, in which case an
    external one cannot take precedence and activate_external_torch() is a
    no-op.

    Deliberately checks where torch actually lives rather than just whether
    it's importable: once an external directory is on sys.path, a frozen app
    can import a torch that is not bundled at all, and calling that
    "bundled" told users their external install had been ignored when it
    hadn't.
    """
    if _bundle_dir() is None:
        return False
    module = sys.modules.get("torch")
    if module is not None:
        return _is_inside_bundle(getattr(module, "__file__", None))
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return False
    return spec is not None and _is_inside_bundle(spec.origin)


def activate_external_torch() -> Path | None:
    """Put the user's torch directory on sys.path, ahead of everything else.

    Returns the directory actually activated, or None. Must run before
    anything imports torch - `main()` calls it first thing.
    """
    if torch_is_bundled():
        return None
    directory = torch_runtime_dir()
    if not directory.is_dir():
        return None
    resolved = str(directory.resolve())
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)
    return directory


# PyTorch wheel indexes for CUDA builds, oldest first. A driver reports the
# newest CUDA it supports, so the right wheel is the newest one at or below
# that.
_CUDA_WHEELS = ((11, 8), (12, 1), (12, 4), (12, 6), (12, 8))


def detect_driver_cuda() -> tuple[int, int] | None:
    """(major, minor) CUDA version this machine's NVIDIA driver supports, or
    None when there's no usable GPU. Reads `nvidia-smi`, which ships with the
    driver on both Windows and Linux."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    # nvidia-smi's header ends with e.g. "CUDA Version: 12.4"
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", completed.stdout or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_torch_variant() -> str:
    """The wheel variant to install on this machine: a CUDA build when a
    supported NVIDIA GPU is present, otherwise "cpu"."""
    driver = detect_driver_cuda()
    if driver is None:
        return "cpu"
    usable = [wheel for wheel in _CUDA_WHEELS if wheel <= driver]
    if not usable:
        return "cpu"
    major, minor = max(usable)
    return f"cu{major}{minor}"


def gpu_status() -> int:
    """Print where torch is coming from, whether the GPU is usable, and the
    exact next command if it isn't. Always exits 0 - it's a report."""
    driver = detect_driver_cuda()
    if driver is None:
        print("GPU:            none detected (no nvidia-smi on PATH)")
    else:
        print(f"GPU:            NVIDIA driver supports CUDA {driver[0]}.{driver[1]}")

    activate_external_torch()

    try:
        import torch
    except ImportError:
        print(f"torch source:   none - nothing installed at {torch_runtime_dir()}")
        print("torch:          not installed")
        print("\nTo enable Cellpose, run:\n  vtea-napari --install-torch")
        print(f"which will install the {detect_torch_variant()} build for this machine.")
        return 0

    # Report where it actually came from, not where it was meant to.
    if torch_is_bundled():
        print("torch source:   bundled in this build (cannot be replaced)")
    else:
        print(f"torch source:   {torch.__file__}")

    cuda_build = getattr(torch.version, "cuda", None)
    print(f"torch:          {torch.__version__} ({'CUDA ' + cuda_build if cuda_build else 'CPU-only'})")
    if cuda_build and torch.cuda.is_available():
        print(f"GPU in use:     yes - {torch.cuda.get_device_name(0)}")
        print("\nCellpose will run on the GPU.")
    elif cuda_build:
        print("GPU in use:     no - this torch is a CUDA build but no GPU is visible to it")
        print("\nCheck your NVIDIA driver, or that this machine has an NVIDIA GPU.")
    elif torch_is_bundled():
        print("GPU in use:     no - this build has CPU-only torch baked in")
        print(
            "\nThe deep-learning download is CPU-only by design. For GPU, use the slim "
            "download instead and run:\n  vtea-napari --install-torch"
        )
    else:
        print("GPU in use:     no - CPU-only torch is installed")
        print(
            f"\nFor GPU, reinstall with:\n  vtea-napari --install-torch {detect_torch_variant()}"
        )
    return 0


def _usable_python(command: list[str]) -> bool:
    """True if `command` is a real interpreter that has pip.

    Guards against Windows' Microsoft Store alias: a stub `python.exe` sits
    in %LOCALAPPDATA%\\Microsoft\\WindowsApps and is on PATH by default, so
    shutil.which() finds it, but running it opens the Store instead of
    executing anything. Checking that pip answers rules it out.
    """
    try:
        completed = subprocess.run(
            [*command, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and "pip" in (completed.stdout or "")


def _host_python() -> list[str] | None:
    """A working interpreter command to run pip with, or None.

    A frozen build has no pip of its own and sys.executable is the app, so
    installing borrows a Python from the machine. Returns a command list
    because Windows' launcher is two tokens (`py -3`).
    """
    if not getattr(sys, "frozen", False):
        return [sys.executable]

    candidates: list[list[str]] = []
    if sys.platform == "win32":
        # The py launcher first: it's the documented way to find Python on
        # Windows, is installed by the python.org installer even when
        # "Add to PATH" was skipped, and never resolves to the Store stub.
        launcher = shutil.which("py")
        if launcher:
            candidates.append([launcher, "-3"])
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found and "WindowsApps" not in found:
            candidates.append([found])

    for candidate in candidates:
        if _usable_python(candidate):
            return candidate
    return None


def install_torch(variant: str = "cpu", *, target: Path | None = None) -> int:
    """pip-install torch + cellpose into the runtime directory.

    `variant` selects the wheel index: "cpu", or a CUDA build like "cu121"
    / "cu124" for GPU acceleration. Returns a process exit code.
    """
    if not (variant == "cpu" or (variant.startswith("cu") and variant[2:].isdigit())):
        print(
            f"Unknown torch variant {variant!r}. Use 'cpu', or a CUDA build such as "
            f"'cu121'/'cu124' - see https://pytorch.org/get-started/locally/ for the "
            f"one matching your driver."
        )
        return 2

    python = _host_python()
    if python is None:
        hint = (
            "Install Python 3.10+ from python.org, then re-run this command."
            if sys.platform == "win32"
            else "Install Python 3.10+, then re-run this command."
        )
        print(
            f"No working Python with pip was found, which this needs in order to "
            f"download PyTorch.\n{hint}\n"
            f"Already have one? Point {TORCH_PATH_ENV} at its site-packages folder "
            f"instead - see docs/GPU_SETUP.md."
        )
        return 2

    directory = target or torch_runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        *python,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(directory),
        "--index-url",
        f"{_PYTORCH_INDEX}/{variant}",
        # cellpose and its non-torch dependencies aren't on the PyTorch
        # index, so let pip fall back to PyPI for those.
        "--extra-index-url",
        "https://pypi.org/simple",
        "torch",
        "cellpose",
    ]
    print(f"Installing torch ({variant}) and cellpose into {directory}")
    print(" ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print(f"\nDone. VTEA will use it automatically from {directory}.")
    return result.returncode
