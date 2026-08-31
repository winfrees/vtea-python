"""Picking a run back up where it stopped.

Inference over four thousand tiles takes hours, and a run that has to start
again because a node was pre-empted, a card fell over, or somebody closed a
laptop is not a run anybody will trust with real data. Every other phase can
afford to be restartable by being fast; this one cannot.

So each tile's result is recorded as it completes, and a resumed run skips
what is already there. The record is append-only JSON Lines rather than a
file rewritten per tile: rewriting is quadratic in the number of tiles, and
a rewrite interrupted halfway is exactly the failure this exists to survive.
A truncated final line - the process died mid-write - is discarded on read,
which costs one tile and keeps the rest.

The manifest is deliberately not the data. The segmented tiles live in the
scratch store, which has to be kept (`ZarrScratch(keep=True)`) for a resume
to mean anything; this records only what was done, so that the two can be
checked against each other rather than assumed to agree.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vtea_core.blocked.reconcile import Fragment, SeamPolicy

MANIFEST_VERSION = 1


class ManifestMismatch(RuntimeError):
    """A manifest that does not describe the run being resumed."""


def plan_signature(plan: Any, policy: SeamPolicy, function_name: str) -> dict[str, Any]:
    """What has to match for a resume to be a resume.

    Tile size and halo change which objects a seam cuts, so a manifest from
    a different plan describes different objects wearing the same tile
    indices. Rather than detect that later as a strange result, it is
    refused here.
    """
    return {
        "shape": list(plan.shape),
        "tile": list(plan.tile),
        "halo": list(plan.halo),
        "function": function_name,
        "policy": policy.to_dict(),
    }


@dataclass
class RunManifest:
    """What a segmentation run has finished, as it finishes it."""

    path: Path
    signature: dict[str, Any]
    completed: dict[tuple[int, ...], list[Fragment]] = field(default_factory=dict)
    next_id: int = 1
    _handle: Any = None

    @property
    def n_completed(self) -> int:
        return len(self.completed)

    def fragments(self) -> list[Fragment]:
        """Every fragment recorded so far, in tile order, so a resumed run
        sees the same list a straight-through one would have built."""
        return [
            fragment
            for tile in sorted(self.completed)
            for fragment in self.completed[tile]
        ]

    def is_done(self, tile_index: Sequence[int]) -> bool:
        return tuple(tile_index) in self.completed

    def record(
        self, tile_index: Sequence[int], fragments: Iterable[Fragment], next_id: int
    ) -> None:
        """Append one finished tile. Flushed immediately - a record that is
        still in a buffer when the process dies is not a record."""
        entry = tuple(int(value) for value in tile_index)
        catalogued = list(fragments)
        self.completed[entry] = catalogued
        self.next_id = int(next_id)
        line = json.dumps(
            {
                "tile": list(entry),
                "next_id": self.next_id,
                "fragments": [fragment.to_dict() for fragment in catalogued],
            }
        )
        handle = self._open()
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> RunManifest:  # noqa: PYI034 - typing.Self needs Python 3.11+, this package supports 3.10
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _open(self):
        if self._handle is None:
            # Held open across the whole run rather than reopened per
            # tile: this is appended to thousands of times, and the
            # manifest itself is the context manager that closes it.
            self._handle = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        return self._handle

    @classmethod
    def start(
        cls, path: str | os.PathLike, signature: dict[str, Any], *, overwrite: bool = False
    ) -> RunManifest:
        """A fresh manifest, or the existing one for this exact run.

        A manifest for a *different* run is refused rather than appended to:
        two runs' tile indices mean different things, and silently mixing
        them would produce a label array that is wrong in a way nothing
        would catch.
        """
        path = Path(os.fspath(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            existing = cls.load(path)
            if existing.signature != signature:
                raise ManifestMismatch(
                    f"{path} records a different run: "
                    f"{_describe_difference(existing.signature, signature)}. Delete it to "
                    f"start again, or resume with the settings it was made with."
                )
            return existing
        path.write_text(
            json.dumps({"version": MANIFEST_VERSION, "signature": signature}) + "\n"
        )
        return cls(path=path, signature=signature)

    @classmethod
    def load(cls, path: str | os.PathLike) -> RunManifest:
        path = Path(os.fspath(path))
        lines = path.read_text().splitlines()
        if not lines:
            raise ManifestMismatch(f"{path} is empty")
        header = json.loads(lines[0])
        if int(header.get("version", 1)) > MANIFEST_VERSION:
            raise ManifestMismatch(
                f"{path} was written in format version {header['version']} and this "
                f"VTEA reads up to {MANIFEST_VERSION}"
            )
        manifest = cls(path=path, signature=header.get("signature", {}))
        for line in lines[1:]:
            try:
                entry = json.loads(line)
            except ValueError:
                # A half-written final line means the process died during
                # that tile. Losing it costs one tile's work and keeps
                # everything before it, which is the whole point.
                break
            manifest.completed[tuple(entry["tile"])] = [
                Fragment.from_dict(item) for item in entry["fragments"]
            ]
            manifest.next_id = int(entry["next_id"])
        return manifest


def _describe_difference(existing: dict[str, Any], wanted: dict[str, Any]) -> str:
    changed = [
        f"{key} was {existing.get(key)!r}, now {wanted.get(key)!r}"
        for key in sorted(set(existing) | set(wanted))
        if existing.get(key) != wanted.get(key)
    ]
    return "; ".join(changed) or "the settings differ"
