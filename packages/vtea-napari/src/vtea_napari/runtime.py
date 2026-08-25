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


def torch_is_bundled() -> bool:
    """True when this build has torch frozen in, in which case an external
    one cannot take precedence and activate_external_torch() is a no-op."""
    if "torch" in sys.modules:
        return True
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return False
    return spec is not None and getattr(sys, "frozen", False)


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


def _host_python() -> str | None:
    """An interpreter to run pip with.

    A frozen build has no pip and sys.executable is the app itself, so
    installing needs a Python from the machine. Most people running a
    scientific imaging tool have one; if not, say so rather than failing
    obscurely.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    for candidate in ("python3", "python"):
        found = shutil.which(candidate)
        if found:
            return found
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
        print(
            "Installing PyTorch needs a Python interpreter on your PATH, and none was "
            "found. Install Python 3.10+ and re-run, or install torch yourself and "
            f"point {TORCH_PATH_ENV} at its site-packages directory."
        )
        return 2

    directory = target or torch_runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        python,
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
