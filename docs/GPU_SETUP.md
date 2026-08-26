# Using a GPU with VTEA (Cellpose)

Cellpose segmentation needs PyTorch. VTEA does not bundle CUDA, so this page
is how you connect VTEA to a GPU-capable PyTorch.

**First, find out which VTEA you have** — the answer is completely different
for each:

| What you have | Go to |
|---|---|
| You downloaded a `.zip` from [Releases](https://github.com/winfrees/vtea-python/releases) and run `vtea-napari.exe` | [Standalone download](#standalone-download) |
| You ran `pip install` / `conda install` into a Python environment | [pip install](#pip-install) |

At any point, this tells you exactly where you stand:

```
vtea-napari --gpu-status
```

```
GPU:            NVIDIA driver supports CUDA 12.4
torch source:   C:\Users\you\AppData\Local\vtea\torch
torch:          2.6.0+cu124 (CUDA 12.4)
GPU in use:     yes - NVIDIA RTX A4000

Cellpose will run on the GPU.
```

---

## Standalone download

Use **`vtea-napari-windows.zip`** (the slim one). The
`-deeplearning` build cannot use a GPU — it has CPU-only PyTorch baked in,
and a bundled PyTorch can never be replaced, so there is no way to add GPU
support to it afterwards.

### Option 1 — let VTEA install PyTorch (recommended)

Open **PowerShell**, `cd` into the unzipped folder, and run:

```powershell
.\vtea-napari.exe --install-torch
```

That checks your NVIDIA driver, picks the matching CUDA build, and installs
PyTorch + Cellpose into `%LOCALAPPDATA%\vtea\torch`. VTEA finds it there
automatically from then on — there is nothing to configure.

To force a specific build instead of auto-detecting:

```powershell
.\vtea-napari.exe --install-torch cu124
```

> **Requires Python on your PATH.** The standalone app has no `pip` of its
> own, so it borrows one. If you see *"Installing PyTorch needs a Python
> interpreter on your PATH"*, either install Python 3.10+ from
> [python.org](https://www.python.org/downloads/) (tick *"Add python.exe to
> PATH"*), or use Option 2.

### Option 2 — point VTEA at an environment you already have

If you already have a conda env or venv with a CUDA PyTorch, reuse it. You
need its `site-packages` path. To find it, activate that environment and
run:

```powershell
python -c "import site; print(site.getsitepackages()[-1])"
```

Typical result: `C:\Users\you\miniconda3\envs\cellpose\Lib\site-packages`

Then point VTEA at it. Set it permanently (survives reboots):

```powershell
setx VTEA_TORCH_PATH "C:\Users\you\miniconda3\envs\cellpose\Lib\site-packages"
```

Close and reopen PowerShell — `setx` only affects new sessions — then start
VTEA and check:

```powershell
.\vtea-napari.exe --gpu-status
```

<details>
<summary>Just this session, or other shells</summary>

PowerShell (current window only):
```powershell
$env:VTEA_TORCH_PATH = "C:\path\to\site-packages"
```

Command Prompt (current window only):
```cmd
set VTEA_TORCH_PATH=C:\path\to\site-packages
```

macOS / Linux:
```bash
export VTEA_TORCH_PATH=/path/to/env/lib/python3.11/site-packages
```

Windows GUI: Start → *"Edit the system environment variables"* →
**Environment Variables…** → under *User variables*, **New…** → name
`VTEA_TORCH_PATH`, value the path above.
</details>

That environment must have a matching Cellpose too (`pip install cellpose`
inside it), since VTEA loads both from there.

---

## pip install

Nothing VTEA-specific is needed — install PyTorch into the **same
environment** as VTEA and it is picked up like any other import.
`VTEA_TORCH_PATH` is not used here; it exists only for the standalone build.

```bash
# in the environment where you installed vtea-napari
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install cellpose
```

Replace `cu124` with the build matching your driver — see
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
or run `vtea-napari --gpu-status`, which names the right one for your
machine.

Verify:

```bash
vtea-napari --gpu-status
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

---

## Which CUDA build do I need?

Run `nvidia-smi`. The top-right corner shows the newest CUDA your driver
supports:

```
| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4  |
```

Install the PyTorch build **at or below** that number — `cu124` here. A
newer PyTorch CUDA build than your driver supports will install but fail at
runtime. `--install-torch` with no argument does this for you.

If `nvidia-smi` isn't found, you either have no NVIDIA GPU or no driver
installed; Cellpose will run on the CPU.

---

## Troubleshooting

**`--gpu-status` says "torch: not installed"** — nothing is installed yet at
the path shown. Run `vtea-napari --install-torch`, or set `VTEA_TORCH_PATH`.

**"this torch is a CUDA build but no GPU is visible to it"** — PyTorch is
correct but cannot reach the GPU. Usually an outdated driver, or a machine
with no NVIDIA GPU. Check `nvidia-smi` works on its own.

**"this build has CPU-only torch baked in"** — you are running the
`-deeplearning` download. Switch to the slim `vtea-napari-<os>.zip` and use
`--install-torch`.

**Set `VTEA_TORCH_PATH` but nothing changed** — if you used `setx`, open a
new terminal; it does not affect the window you typed it in. Confirm with
`echo $env:VTEA_TORCH_PATH` (PowerShell). The path must be the
`site-packages` directory itself — the folder that *contains* a `torch`
folder — not the environment root.

**Cellpose still fails on first use** — Cellpose downloads its pretrained
weights the first time it segments, which needs internet access. This is
separate from PyTorch and applies to every VTEA variant.
