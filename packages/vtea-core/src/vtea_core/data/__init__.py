"""VolumeDataset abstraction: in-memory (NumPy) and chunked (Dask/Zarr) volumetric data.

Ports vtea.dataset / vtea.dataset.volume / vtea.partition from the Java codebase.
Dask's map_blocks/map_overlap replace the hand-written Chunk/VolumePartitioner/
ChunkIterator; ObjectStitcher's cross-chunk object-merge logic still needs a
direct port (see PORT_PLAN.md, Phase 1).
"""

from vtea_core.data.objects import object_ids, object_intensity_values, object_pixel_indices
from vtea_core.data.volume import ChunkedVolumeDataset, InMemoryVolumeDataset, VolumeDataset

__all__ = [
    "VolumeDataset",
    "InMemoryVolumeDataset",
    "ChunkedVolumeDataset",
    "object_ids",
    "object_pixel_indices",
    "object_intensity_values",
]
