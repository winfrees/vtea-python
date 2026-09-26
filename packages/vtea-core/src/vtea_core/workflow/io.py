"""Reading and writing a protocol as JSON (`*.vtea.json`).

Tier 1 of docs/SAVING_AND_ARCHIVING.md: everything needed to re-run an
analysis on the next image, and nothing that can be recomputed. A protocol
is the two step stacks the builder shows (processing, then analysis), how
to read the image's axes, the voxel size, and optionally the gates drawn on
the results - the one part of an analysis that is a judgement rather than a
setting, and that a class rule may refer to by name.

Plain, versioned JSON for the same reasons as the gate file
(`vtea_core.gates.io`): it opens anywhere, diffs in version control and can
be mailed between labs. It is also why nothing here is ever `eval`'d or
unpickled. A parameter that is not plain data - a trained model, an
arbitrary object - is refused on save with the step and parameter named,
rather than written as something that half-loads later.

What the file deliberately does not carry: results (they are recomputed),
regions painted on a napari Labels layer (image gates are layer data, not
settings), and hand-corrected association links (those save through
`vtea_core.objects.save_associations`).
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from vtea_core.context.spec import ContextSpec
from vtea_core.data.spacing import Spacing
from vtea_core.gates.gate import GateSet
from vtea_core.gates.io import gate_set_from_dict, gate_set_to_dict
from vtea_core.workflow.pipeline import Pipeline, Step
from vtea_core.workflow.registry import STEP_REGISTRY

# Bumped only for a breaking layout change; a reader checks it so a file
# from a future version fails clearly instead of half-loading.
# 2 adds the `context` section; a protocol without one is still written as 1.
PROTOCOL_FORMAT_VERSION = 2

PROTOCOL_SUFFIX = ".vtea.json"

# Packages whose version moves the numbers a protocol produces. Recorded so
# that "the counts changed" can be traced to "scikit-image changed" rather
# than to the protocol.
ENVIRONMENT_PACKAGES = (
    "vtea-core",
    "vtea-napari",
    "numpy",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "pandas",
    "dask",
    "zarr",
    "tifffile",
    "duckdb",
    "umap-learn",
    "python-igraph",
    "leidenalg",
    "torch",
    "cellpose",
)

# Hashing a source image is a checksum over every byte of it. Worth it for
# the files people email; not for a 40 GB acquisition every time somebody
# clicks Save.
DEFAULT_HASH_LIMIT_BYTES = 2 * 1024**3


class ProtocolError(ValueError):
    """A protocol that cannot be written, or a file that is not one this
    VTEA can load."""


@dataclass
class Protocol:
    """One saved analysis: the steps, and how to read the image they ran on.

    `gates` maps a table name ("Objects", or a cells table) to the gates
    drawn on it. `source` identifies the image the protocol was built
    against (see `describe_source`); it is a record, not a requirement - a
    protocol is meant to be re-run on the next image.
    """

    processing: Pipeline = field(default_factory=Pipeline)
    analysis: Pipeline = field(default_factory=Pipeline)
    channel_axis: int | None = None
    z_axis: int | None = None
    spacing: Spacing | None = None
    measure_every_segmentation: bool = True
    gates: dict[str, GateSet] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    # The hierarchy of levels the analysis moves between - subcellular,
    # cellular, neighbourhoods of any depth - as definitions, rebuilt from
    # each run's results (see vtea_core.context).
    context: ContextSpec = field(default_factory=ContextSpec)
    # Filled in on save and read back on load; informational only.
    created: str = ""
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> list[Step]:
        return list(self.processing.steps) + list(self.analysis.steps)


# -- parameters -------------------------------------------------------------


def encode_value(value: Any, *, where: str = "parameter") -> Any:
    """A parameter value as JSON-safe data, or ProtocolError.

    Plain scalars, strings, lists and string-keyed dicts pass through.
    Tuples and arrays are tagged so they come back as what they were: a
    function that checks `isinstance(value, tuple)` should not see a list
    after a round trip, and a step's settings signature should not change
    just because it was saved.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"$tuple": [encode_value(item, where=where) for item in value]}
    if isinstance(value, list):
        return [encode_value(item, where=where) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise ProtocolError(f"{where}: an object array cannot be saved in a protocol")
        return {"$array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, Spacing):
        return {"$spacing": value.to_dict()}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ProtocolError(f"{where}: only dicts with string keys can be saved")
        if any(key.startswith("$") for key in value):
            raise ProtocolError(f"{where}: dict keys starting with '$' are reserved")
        return {key: encode_value(item, where=where) for key, item in value.items()}
    raise ProtocolError(
        f"{where} is a {type(value).__name__}, which a protocol cannot carry - "
        "only numbers, text, lists, tuples, arrays and voxel sizes are saved"
    )


def decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if isinstance(value, dict):
        if "$tuple" in value:
            return tuple(decode_value(item) for item in value["$tuple"])
        if "$array" in value:
            return np.asarray(value["$array"], dtype=value.get("dtype"))
        if "$spacing" in value:
            return Spacing.from_dict(value["$spacing"])
        return {key: decode_value(item) for key, item in value.items()}
    return value


# -- steps ------------------------------------------------------------------


def step_to_dict(step: Step) -> dict[str, Any]:
    where = f"step '{step.name or step.function_name}'"
    return {
        "name": step.name,
        "category": step.category,
        "function": step.function_name,
        "params": {
            key: encode_value(value, where=f"{where}, parameter '{key}'")
            for key, value in step.params.items()
        },
        "input_keys": dict(step.input_keys),
        "output_key": step.output_key,
        "channel": step.channel,
        # Empty means "every measured feature", and is kept empty rather than
        # expanded: re-run on an image with a fourth channel, the step should
        # use the features that image has.
        "features": list(step.features),
        "auto_for": step.auto_for,
        "comment": step.comment,
    }


def step_from_dict(data: dict[str, Any]) -> Step:
    category = data["category"]
    function_name = data["function"]
    if function_name not in STEP_REGISTRY.get(category, {}):
        raise ProtocolError(
            f"step '{data.get('name') or function_name}' uses {category}.{function_name}, "
            "which this VTEA does not have - the protocol may come from a newer version"
        )
    return Step(
        category=category,
        function_name=function_name,
        params={key: decode_value(value) for key, value in data.get("params", {}).items()},
        input_keys=dict(data.get("input_keys", {})),
        output_key=data.get("output_key", "result"),
        comment=data.get("comment", ""),
        channel=data.get("channel"),
        name=data.get("name", ""),
        features=list(data.get("features", [])),
        auto_for=data.get("auto_for", ""),
    )


# -- provenance -------------------------------------------------------------


def capture_environment(packages=ENVIRONMENT_PACKAGES) -> dict[str, Any]:
    """Python and the installed versions of the packages that decide what a
    protocol computes. Missing packages are left out rather than listed as
    absent, so the record reads as what actually ran."""
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {"python": platform.python_version(), "packages": versions}


def describe_source(
    path: str | Path | None = None,
    *,
    name: str | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: Any = None,
    hash_limit: int | None = DEFAULT_HASH_LIMIT_BYTES,
) -> dict[str, Any]:
    """What a protocol was built against: enough to tell later whether a
    re-run used the same data.

    The SHA-256 is computed when `path` is a file no larger than
    `hash_limit` (None = always). Above that the size is recorded instead,
    and the record says the checksum was skipped rather than leaving the
    reader to wonder.
    """
    source: dict[str, Any] = {}
    if name:
        source["name"] = name
    if shape is not None:
        source["shape"] = [int(size) for size in shape]
    if dtype is not None:
        source["dtype"] = str(np.dtype(dtype))
    if path is not None:
        path = Path(path)
        source["path"] = str(path)
        if path.is_file():
            size = path.stat().st_size
            source["bytes"] = size
            if hash_limit is None or size <= hash_limit:
                source["sha256"] = file_sha256(path)
            else:
                source["sha256"] = None
                source["sha256_skipped"] = f"file larger than {hash_limit} bytes"
    return source


def file_sha256(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


# -- protocols --------------------------------------------------------------


def protocol_to_dict(protocol: Protocol, *, environment: dict[str, Any] | None = None) -> dict:
    data = {
        # Version 2 only when there is a context section to read: a
        # protocol without one is still one every earlier VTEA can open,
        # while one with it must not open in a VTEA that would silently drop
        # its levels.
        "vtea_protocol_version": PROTOCOL_FORMAT_VERSION if protocol.context else 1,
        "created": protocol.created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": dict(protocol.source),
        "axes": {"channel_axis": protocol.channel_axis, "z_axis": protocol.z_axis},
        "spacing": None if protocol.spacing is None else protocol.spacing.to_dict(),
        "measure_every_segmentation": bool(protocol.measure_every_segmentation),
        "processing": [step_to_dict(step) for step in protocol.processing.steps],
        "analysis": [step_to_dict(step) for step in protocol.analysis.steps],
        "gates": {name: gate_set_to_dict(gates) for name, gates in protocol.gates.items()},
        "environment": capture_environment() if environment is None else environment,
    }
    if protocol.context:
        data["context"] = _encode_context(protocol.context)
    return data


def _encode_context(context: ContextSpec) -> dict[str, Any]:
    data = context.to_dict()
    for level in data["levels"]:
        for section in ("build", "measure", "classify"):
            level[section] = {
                key: encode_value(value, where=f"level '{level['name']}', {section} '{key}'")
                for key, value in level[section].items()
            }
    return data


def _decode_context(data: dict[str, Any] | None) -> ContextSpec:
    if not data:
        return ContextSpec()
    for level in data.get("levels", []):
        for section in ("build", "measure", "classify"):
            level[section] = {
                key: decode_value(value) for key, value in level.get(section, {}).items()
            }
    try:
        return ContextSpec.from_dict(data)
    except (KeyError, ValueError) as error:
        raise ProtocolError(f"the protocol's context section cannot be read: {error}") from error


def protocol_from_dict(data: dict[str, Any]) -> Protocol:
    if not isinstance(data, dict) or "vtea_protocol_version" not in data:
        raise ProtocolError("not a VTEA protocol file (no 'vtea_protocol_version')")
    version = data["vtea_protocol_version"]
    if version > PROTOCOL_FORMAT_VERSION:
        raise ProtocolError(
            f"protocol version {version} is newer than this VTEA understands "
            f"({PROTOCOL_FORMAT_VERSION}); upgrade vtea-core to open it"
        )
    axes = data.get("axes", {})
    channel_axis = axes.get("channel_axis")
    spacing = data.get("spacing")
    return Protocol(
        processing=Pipeline(
            [step_from_dict(entry) for entry in data.get("processing", [])],
            channel_axis=channel_axis,
        ),
        analysis=Pipeline(
            [step_from_dict(entry) for entry in data.get("analysis", [])],
            channel_axis=channel_axis,
        ),
        channel_axis=channel_axis,
        z_axis=axes.get("z_axis"),
        spacing=None if spacing is None else Spacing.from_dict(spacing),
        measure_every_segmentation=bool(data.get("measure_every_segmentation", True)),
        gates={name: gate_set_from_dict(entry) for name, entry in data.get("gates", {}).items()},
        source=dict(data.get("source", {})),
        context=_decode_context(data.get("context")),
        created=data.get("created", ""),
        environment=dict(data.get("environment", {})),
    )


def save_protocol(protocol: Protocol, path: str | Path) -> Path:
    """Write `protocol` as JSON. Everything is encoded before anything is
    written, so a protocol that cannot be saved leaves no half-written file
    behind."""
    text = json.dumps(protocol_to_dict(protocol), indent=2)
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path


def load_protocol(path: str | Path) -> Protocol:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProtocolError(f"{path} is not valid JSON: {error}") from error
    return protocol_from_dict(data)
