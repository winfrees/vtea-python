# vtea-napari

napari plugin GUI for VTEA (Volumetric Tissue Exploration and Analysis) — a
thin UI layer over [`vtea-core`](../vtea-core). napari is the closest Python
analog to the ImageJ/Fiji viewer the Java application plugs into today: it's
Qt-based, has native 3D volume rendering, and an active plugin ecosystem.

See [`/docs/PORT_PLAN.md`](../../docs/PORT_PLAN.md) in the repo root for the
full porting plan, including the still-open scope decision on how to handle
the pipeline/protocol builder (the largest, most bespoke piece of the
original Java UI).

## Status

Phase 0 (foundations) — package skeleton and an empty npe2 manifest only. No
widgets have been ported yet; that's Phase 4.
