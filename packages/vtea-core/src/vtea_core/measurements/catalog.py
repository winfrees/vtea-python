"""What each column of the measurement table is, and how it was produced.

A table column called `mean_ch2` or `pca_1_1` is opaque on its own. To
anyone reading the analysis later - a collaborator, a reviewer, the person
who ran it six months on - the questions are always the same: what was
measured, on which channel, of which segmentation, by which step, with what
parameters; and for a derived feature, which features went into it.

The catalog answers those per column. It is built as steps run, saved as
plain JSON with the protocol, and rendered as the `data_dictionary.csv` the
publication bundle carries (see docs/SAVING_AND_ARCHIVING.md) - which is
what makes a deposited table self-describing rather than a wall of unlabelled
numbers.

Nothing here infers provenance after the fact from column names alone: the
step that produced a column is recorded at the moment it produces it, and
only the *measurement* half of a name (mean vs. count, and which channel) is
parsed, because that part the naming scheme genuinely does encode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from vtea_core.measurements.regionprops import (
    GEOMETRY_COLUMNS,
    NON_FEATURE_COLUMNS,
    NON_FEATURE_PREFIXES,
    parse_feature_name,
)

CATALOG_FORMAT_VERSION = 1

# What a column is, as opposed to what it measures.
IDENTIFIER = "identifier"  # object_id - a row label, not a feature
GEOMETRY = "geometry"  # count, centroid-* - the object's shape and place
INTENSITY = "intensity"  # measured off the image, on a named channel
DERIVED = "derived"  # computed from other features (PCA, clusters)

# Best-effort units for the built-in measurements. Intensity is left as
# arbitrary units: a raw detector count means nothing absolute without
# calibration VTEA doesn't have.
_UNITS = {
    "count": "voxels",
    "mean": "a.u.",
    "min": "a.u.",
    "max": "a.u.",
    "sum": "a.u.",
    "stddev": "a.u.",
    "threshold_mean": "a.u.",
}


@dataclass
class FeatureDescriptor:
    """One column of the measurement table, and where it came from."""

    name: str
    kind: str
    measurement: str = ""
    channel: int | None = None
    # The named segmentation whose objects these rows are - which is the
    # thing two measurement steps in one protocol differ by.
    segmentation: str = ""
    produced_by: str = ""  # step name
    function: str = ""  # "category.function_name"
    params: dict[str, Any] = field(default_factory=dict)
    # For a derived feature: exactly which features were fed to the step
    # that produced it. This is the record that makes a PCA reproducible.
    source_features: list[str] = field(default_factory=list)
    units: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDescriptor:
        known = {key: data[key] for key in data if key in cls.__dataclass_fields__}
        return cls(**known)


def classify_column(name: str) -> tuple[str, str, int | None]:
    """(kind, measurement, channel) for a measured column, from its name.

    Only used for columns a measurement step produced - a derived feature is
    described from its step, not guessed at.
    """
    if name in NON_FEATURE_COLUMNS:
        return IDENTIFIER, name, None
    if name.startswith(NON_FEATURE_PREFIXES):
        axis = name.split("-", 1)[1]
        return GEOMETRY, f"centroid along axis {axis}", None
    measurement, channel = parse_feature_name(name)
    if measurement in GEOMETRY_COLUMNS:
        return GEOMETRY, measurement, channel
    return INTENSITY, measurement, channel


class FeatureCatalog:
    """name -> FeatureDescriptor, in the order the features were added."""

    def __init__(self, descriptors: list[FeatureDescriptor] | None = None):
        self._by_name: dict[str, FeatureDescriptor] = {}
        for descriptor in descriptors or []:
            self.add(descriptor)

    def add(self, descriptor: FeatureDescriptor) -> FeatureDescriptor:
        self._by_name[descriptor.name] = descriptor
        return descriptor

    def get(self, name: str) -> FeatureDescriptor | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return list(self._by_name)

    def drop_missing(self, columns) -> None:
        """Forget features that are no longer in the table - a re-run with a
        different measurement step leaves the old entries stale, and a stale
        entry is worse than a missing one because it looks authoritative."""
        keep = set(columns)
        self._by_name = {name: d for name, d in self._by_name.items() if name in keep}

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    # -- recording --------------------------------------------------------

    def record_measured(
        self,
        columns,
        *,
        produced_by: str = "",
        function: str = "",
        params: dict[str, Any] | None = None,
        segmentation: str = "",
    ) -> list[FeatureDescriptor]:
        """Describe the columns a measurement step just produced."""
        recorded = []
        for name in columns:
            kind, measurement, channel = classify_column(name)
            recorded.append(
                self.add(
                    FeatureDescriptor(
                        name=name,
                        kind=kind,
                        measurement=measurement,
                        channel=channel,
                        segmentation=segmentation,
                        produced_by=produced_by,
                        function=function,
                        params=dict(params or {}),
                        units=_UNITS.get(measurement, ""),
                    )
                )
            )
        return recorded

    def record_derived(
        self,
        columns,
        *,
        produced_by: str = "",
        function: str = "",
        params: dict[str, Any] | None = None,
        source_features=(),
        segmentation: str = "",
        measurement: str = "",
    ) -> list[FeatureDescriptor]:
        """Describe the columns a clustering or reduction step produced,
        recording which features were fed to it."""
        sources = list(source_features)
        return [
            self.add(
                FeatureDescriptor(
                    name=name,
                    kind=DERIVED,
                    measurement=measurement or _derived_measurement(function, name),
                    segmentation=segmentation,
                    produced_by=produced_by,
                    function=function,
                    params=dict(params or {}),
                    source_features=sources,
                    units="",
                )
            )
            for name in columns
        ]

    # -- output -----------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """The data dictionary: one row per feature, in the column order a
        reader wants (what it is, then where it came from)."""
        rows = [
            {
                "column": d.name,
                "kind": d.kind,
                "measurement": d.measurement,
                "channel": "" if d.channel is None else d.channel,
                "segmentation": d.segmentation,
                "produced_by": d.produced_by,
                "function": d.function,
                "params": _render_params(d.params),
                "source_features": ", ".join(d.source_features),
                "units": d.units,
            }
            for d in self
        ]
        columns = [
            "column",
            "kind",
            "measurement",
            "channel",
            "segmentation",
            "produced_by",
            "function",
            "params",
            "source_features",
            "units",
        ]
        return pd.DataFrame(rows, columns=columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vtea_feature_catalog_version": CATALOG_FORMAT_VERSION,
            "features": [d.to_dict() for d in self],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureCatalog:
        version = data.get("vtea_feature_catalog_version")
        if version is not None and version > CATALOG_FORMAT_VERSION:
            raise ValueError(
                f"feature catalog version {version} is newer than this VTEA "
                f"understands ({CATALOG_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
        return cls([FeatureDescriptor.from_dict(entry) for entry in data.get("features", [])])


def _derived_measurement(function: str, name: str) -> str:
    category = function.split(".", 1)[0]
    if category == "clustering":
        return "cluster assignment"
    if category == "reduction":
        return "reduced dimension"
    return name


def _render_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
