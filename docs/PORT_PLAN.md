# VTEA Python Port — Implementation Plan

> Ported from the Java source application at
> [winfrees/volumetric-tissue-exploration-analysis](https://github.com/winfrees/volumetric-tissue-exploration-analysis).
> This document is the canonical, living copy of the plan; update it here as
> phases complete.

## Goal and scope

Fully replace the Java/ImageJ1/SciJava VTEA application with a Python-native equivalent, distributed as a **napari plugin** backed by a standalone, headless-usable **core analysis library**. Java is retired once parity is reached; it is not kept as a permanent hybrid host (the codebase already tried that pattern for deep learning — see "Why not extend the existing Py4J bridge" below).

**Primary motivation:** several of VTEA's Java dependencies are effectively unmaintained or awkward wrappers around ecosystems that are Python-native and actively developed — JavaCPP/PyTorch bindings for the 3D VAE/CNN stack, a subprocess+socket (Py4J) bridge to run Cellpose (itself a Python package), Renjin embedding an old R interpreter for a handful of plotting calls, and JFreeChart/XChart/vioplot in place of matplotlib/plotly. Porting removes these translation layers instead of adding more of them.

## Current state (facts, from codebase inventory)

- **451 Java files, ~109K LOC**, Maven build, Java 8, ~30 packages under `src/main/java/vtea/*`, `vteaobjects`, `vteaexploration`.
- **Entry point:** `vtea._vtea` — legacy ImageJ1 `PlugIn` (its `@Plugin` SciJava annotation is commented out, so it is *not* SciJava-discovered; ImageJ1's `run()` is the real bootstrap). Registered via `src/main/resources/plugins.config`.
- **Plugin/extension architecture:** 13 `vtea.services` classes, each binding a SciJava `PluginService` lookup to an extension-point interface (`Segmentation`, `FeatureProcessing`, `Measurements`, `Morphology`, `LUT`, `GateMath`, `PlotMaker`, `Processor`, `Workflow`, `FileType`, `ImageProcessing`, `NeighborhoodMeasurements`). This is the registry that populates every algorithm dropdown in the UI.
- **Largest packages:** `vtea.objects` (51 files, ~15.5K LOC — segmentation engine, ~15 methods including LayerCake3D/kD-tree, FloodFill3D, MorphoLibJ/ImgLib2 connected components, Cellpose/DeepImageJ, STAPLE), `vtea.protocol` (63 files, ~15.5K LOC — the block-based visual pipeline builder), `vtea.deeplearning` (42 files, ~15.5K LOC — two *parallel* deep-learning integrations, see below), `vtea.exploration` + `vteaexploration` (116 files, ~25K LOC combined — the interactive plotting/gating workbench, `MicroExplorer` main window).
- **Deep learning has two independent paths today:** (1) a Py4J bridge (`CellposeInterface`) that launches `python/cellpose_server.py` as a subprocess and calls it over a `py4j.GatewayServer` socket, with its own restart/backoff and GPU-OOM detection logic; (2) a from-scratch Java 3D VAE/CNN stack on JavaCPP `pytorch-platform`/`cuda-platform` bindings, with its own Swing training UI. Neither talks to the other.
- **Persistence:** `vtea.jdbc.H2DatabaseEngine` — in-memory H2 (`jdbc:h2:mem:VTEADB`), two tables (`MEASUREMENTS`, `OBJECTS`) as the session-scoped store for per-object features. Not disk-persisted by default.
- **R/Renjin:** minimal — `vtea.renjin` (149 LOC) only generates an R color-palette string; despite `ggplot2`/`gplots`/`vioplot` CRAN deps in `pom.xml`, no substantial R usage was found elsewhere.
- **Large-volume support (VTEA 2.0, in progress):** `vtea.dataset.volume` (`VolumeDataset`/`ImagePlusVolumeDataset`/`ZarrVolumeDataset`), `vtea.io.zarr`, `vtea.partition` (`Chunk`, `VolumePartitioner`, `ChunkIterator`, `ObjectStitcher` using a Smile kD-tree for boundary object merging), `vtea.objects.AbstractChunkedSegmentation`. Per the source doc's own "Remaining Work" section, this is partially implemented, not finished — treat as aspirational.
- Two sample datasets already sit at repo root (`AQtest_human_crop.tif`, `C1-IU_VTEA_ExampleData_001.tif`) — useful as the seed for a parity-test golden dataset.

### Why not just extend the existing Py4J bridge (hybrid path)

That's the architecture already used for Cellpose, and the codebase shows its cost first-hand: a subprocess launcher, a socket gateway, JSON parameter marshaling, manual byte-array (de)serialization of images in both directions, and bespoke restart/OOM-recovery logic — all to call a library that's a native `pip install` away. Multiplying that pattern across segmentation, clustering, DR, and plotting would mean permanently maintaining two runtimes, two dependency trees, and a marshaling layer between them, for a codebase whose GUI, algorithms, and glue code all need substantial rework regardless. A full port removes the bridge instead of scaling it.

## Target architecture

Two packages:

- **`vtea-core`** — pure Python library, no GUI dependency. Data model, I/O, segmentation, features, clustering/DR, gating, classification. Usable headless from scripts/Jupyter/CLI/HPC.
- **`vtea-napari`** — napari plugin (dock widgets + `npe2` manifest) that is a thin UI layer over `vtea-core`. napari is Qt-based (PyQt5/PySide2) and is the closest Python analog to the ImageJ/Fiji viewer VTEA plugs into today, with an active plugin ecosystem and native 3D volume rendering.

### Why deep learning isn't a separate module

`vtea.deeplearning` is its own 42-file Java package because it had to be: Cellpose ran over a Py4J subprocess bridge, and the native VAE/CNN stack went through JavaCPP's PyTorch bindings — neither fit the normal in-JVM call pattern the rest of VTEA used, so isolating them into one package was a plumbing necessity. In Python, `torch` and `cellpose` are ordinary imports with the same call mechanics as `scikit-image` or `scikit-learn`; the reason for the isolation is gone. So the port folds deep learning into the domains it actually belongs to instead of resurrecting the Java module boundary:

- **Cellpose** → `vtea_core.segmentation.cellpose_segmentation()`, another way to go from intensity volume to label mask, next to `label_components`/`watershed_split`.
- **DeepImageJ's generic model inference** → `vtea_core.segmentation.model_inference()`, via `bioimageio.core` (that's what DeepImageJ was used for in VTEA).
- **The native VAE/CNN classification stack** → a new `vtea_core.classification` module, parallel to `clustering`/`reduction` — it's supervised/representation-learning classification of segmented objects, conceptually distinct from both (clustering is unsupervised grouping; classification here is trained per-object labeling), so it gets its own module rather than being force-fit into an existing one. A `class_map()` utility (same label-remap pattern `segmentation.filter_by_size()` already uses) maps predictions back onto the label image.

What *does* stay isolated: the heavy dependencies (`torch`, `cellpose`, `bioimageio-core`) stay behind the `deeplearning` extra in `pyproject.toml`, so `pip install vtea-core` doesn't force a multi-GB PyTorch install. That's a packaging concern, independent of where the code lives.

### Dependency mapping

| Java (today) | Python (target) | Notes |
|---|---|---|
| ImageJ1/ImageJ2, SciJava plugin framework | napari + `npe2`/`stevedore`/entry-points registry | Extension points (segmentation, clustering, DR, morphology, LUT, plot makers, workflows) become entry-point groups, same role as the 13 `vtea.services` classes |
| ImgLib2 (n-dim images) | NumPy + Dask arrays, `xarray` for labeled axes | |
| N5 / Zarr, `vtea.io.zarr`, `vtea.partition` | `zarr-python`, `dask.array` (`map_blocks`/`map_overlap`) | Dask's built-in chunking/overlap replaces most of the hand-written `Chunk`/`VolumePartitioner`/`ChunkIterator`; only `ObjectStitcher`'s cross-chunk object-merge logic needs a genuine port |
| Bio-Formats / OME-TIFF import | `bioio` (or `aicsimageio`) for proprietary formats, `tifffile` for TIFF/OME-TIFF, `ome-zarr-py` | `bioio` still uses a JVM under the hood for exotic vendor formats (Zeiss CZI, Leica LIF, etc.) via `scyjava` — that's a transparent runtime dependency, not maintained application code |
| MorphoLibJ, ImgLib2 connected components | `scikit-image` (`morphology`, `segmentation.watershed`, `measure.label`), `scipy.ndimage` | |
| Smile (KMeans/GMM/hierarchical/kD-tree), la4j | `scikit-learn` (`KMeans`, `GaussianMixture`, `AgglomerativeClustering`), `scipy.spatial.cKDTree` | X-Means/G-Means/deterministic annealing have no direct sklearn equivalent — port the BIC/AIC model-selection logic directly |
| `tsne` library, Isomap, Laplacian Eigenmap | `scikit-learn` (`TSNE` or `openTSNE`, `Isomap`, `SpectralEmbedding`), `umap-learn` (new option) | sklearn already ships Isomap and spectral embedding built-in |
| JFreeChart, XChart, vioplot | `matplotlib`/`seaborn` (violin plots), embedded in Qt dock widgets; `plotly` optional for interactive | |
| Swing (`MicroExplorer`, `ProtocolManagerMulti`, gate manager, morphology dialogs, plot windows) | `napari` dock widgets, `magicgui`, raw `qtpy` (PyQt5/PySide2) | See "Highest-risk area" below re: the protocol builder |
| H2 (in-memory JDBC) | `DuckDB` (embedded, columnar, SQL, native Arrow/pandas interop) | Backs the `MEASUREMENTS`/`OBJECTS` tables; also enables on-disk persistence if wanted later |
| Renjin/R (color palette only) | `matplotlib`/`seaborn` colormaps | Drop the R dependency entirely — usage found is a single palette string |
| JavaCPP PyTorch bindings (3D VAE/CNN) | native `torch`, in `vtea_core.classification` | Removes an entire binding layer; direct access to `torch.compile`, mixed precision, model zoo. Lands in `classification`, not a separate `deeplearning` module — see "Why deep learning isn't a separate module" above |
| Py4J bridge + `python/cellpose_server.py` subprocess | in-process `cellpose` import, in `vtea_core.segmentation` | Deletes `CellposeInterface`, the subprocess/socket plumbing, and the bridge script entirely |
| "DeepImageJ" generic model inference | `bioimageio.core`, in `vtea_core.segmentation` | The actual current successor to DeepImageJ, same BioImage Model Zoo spec |
| JNI stub (`HelloJNI`, unused) | — | Drop, not functionally wired in today |

## Protocol builder: Option A (decided)

`vtea.protocol` is 63 files / ~15.5K LOC of pipeline-building Swing UI. Originally flagged as the highest-risk area pending a scope call between a full visual clone (Option A) and a lighter step-list UI (Option B) — **decided: Option A**, a fully functional GUI in napari.

Scoping this against the actual `ProtocolManagerMulti`/`blockstepgui` source corrected an earlier assumption: it is **not** a free-form node-graph editor. `grep` for `TransferHandler`/`DragSource`/`DropTarget`/drag-gesture code across `vtea.protocol` turns up nothing — the layout is a plain `FlowLayout`. There's no wire-based connection UI and no drag-to-reorder; the protocol is an ordered, numbered stack of step cards (process name, parameter-summary comment, thumbnail preview, Edit/Delete buttons) built by adding steps from a category menu (`ExplorationStepBlockGUI`/`FeatureStepBlockGUI`/`MeasurementStepBlockGUI`/`MorphologyStepBlockGUI`/`ObjectStepBlockGUI`/`ProcessStepBlockGUI`), executed top-to-bottom. That's a materially smaller/more tractable Option A than a general node editor would have been, and shapes the Python design:

- **`vtea_core.workflow`** (headless, no Qt): `Step`/`Pipeline` — the same engine whether driven from the GUI or a script/notebook, matching `vtea-core`'s headless-usable design goal. A step registry maps category → available functions, drawing on the real `vtea_core.segmentation`/`measurements`/`clustering`/`reduction`/`gates`/`imageprocessing`/`classification` functions built in Phases 2-3.
- **`vtea-napari`**: a dock widget rendering the step stack as cards (matching the Java layout's information, not its Swing implementation), an "Add Step" category menu, and an Edit dialog. The Edit dialog is `magicgui`-generated directly from each step function's type-hinted signature, rather than hand-built per-algorithm forms (`MicroBlockSetup`'s Java equivalent) — another case of a modern library absorbing what used to be bespoke code.
- Saved Java workflow XML still needs an import converter (unchanged from the original plan) so existing user pipelines aren't stranded — tracked as a follow-up, not blocking the GUI itself.

## Phased roadmap

| Phase | Content | Est. effort |
|---|---|---|
| **0. Foundations & parity harness** | `vtea-core`/`vtea-napari` package skeletons, CI, dependency choices locked in. Build a golden-dataset regression harness: run today's Java pipeline against the two sample TIFFs (already in-repo) and any other representative datasets, capture segmentation masks, feature tables, and cluster/DR outputs as fixtures for every later phase to diff against. | 2–3 weeks |
| **1. Core data model & I/O** | `VolumeDataset` (NumPy in-memory + Dask/Zarr chunked) replacing `vtea.dataset`+`vtea.partition`+`vtea.io.zarr`; object model as labeled arrays + DuckDB/pandas measurement tables replacing `vteaobjects.MicroObject`+H2; readers via `bioio`/`tifffile`/`ome-zarr-py`. | 3–4 weeks |
| **2. Algorithm core** (largest phase) | Segmentation (~15 methods), feature/measurement extraction (`regionprops_table`-based), clustering (KMeans/GMM/hierarchical via sklearn; X-Means/G-Means/annealing ported directly with their BIC/AIC logic), DR (PCA/t-SNE/Isomap/Laplacian Eigenmap), gating, spatial stats, image preprocessing. | 6–10 weeks |
| **3. Deep learning consolidation** | Not a separate `deeplearning` module (see "Why deep learning isn't a separate module") — lands in the domains it belongs to: `cellpose_segmentation()`/`model_inference()` (`bioimageio.core`) in `vtea_core.segmentation`; a new `vtea_core.classification` module (native PyTorch) for the VAE/CNN classification work, replacing the JavaCPP stack. | 3–5 weeks |
| **4. napari plugin (GUI)** | `vtea_core.workflow` (headless Step/Pipeline engine + step registry); the protocol builder as a step-stack dock widget with `magicgui`-generated Edit dialogs (Option A, see above - scoped smaller than originally estimated once the actual Java UI turned out to be a step stack, not a node graph); dock widgets recreating `MicroExplorer` (plots via matplotlib/vispy); gate manager (`PolygonSelector`/`LassoSelector` or vispy-based interactive gating); LUTs (largely free via napari's built-in colormaps); heatmap/violin plot widgets. | 5–8 weeks |
| **5. Parity validation & cutover** | Run the Phase 0 golden-dataset suite end-to-end: segmentation IoU, feature-table numeric diffs, cluster-assignment ARI, against Java outputs. Beta with real users. Docs + workflow-XML migration converter. | 2–4 weeks |
| **6. Decommission Java** | Archive/tag the Java codebase, update the Fiji update site listing to point at the new pip/conda package and napari plugin index. | 1 week |

**Total: roughly 22–35 engineer-weeks (~5–8 months at one senior engineer, less with parallelism across phases 2–4 once Phase 1 lands).**

## Other risks / open questions

- **Numerical parity for custom algorithms** (X-Means, G-Means, deterministic annealing, kD-tree boundary stitching) — these have no drop-in library replacement and must be ported logic-for-logic, then validated against the Phase 0 fixtures.
- **Chunked large-volume behavior** — Dask's overlap/stitching semantics differ from the hand-rolled `Chunk`/`ObjectStitcher` system; validate object counts/boundaries match on a volume too large for RAM before trusting it in production.
- **ImageJ macro compatibility** — `vtea.imageprocessing.builtin.IJMacro` lets users embed arbitrary ImageJ1 macros in a pipeline step. Decide explicitly: drop macro support (replace with scikit-image equivalents), or keep an optional `pyimagej` bridge for legacy macro compatibility during the transition window.
- **Format coverage** — confirm `bioio`'s JVM-backed readers cover every vendor format current users rely on before dropping Bio-Formats from the primary code path.
- **User pipeline continuity** — saved `.xml` workflow/protocol files from the Java app should have a conversion path into the new format (called out in Phase 5) so existing collaborators aren't blocked mid-migration.

## Status

Phases 0–3 are done (see `packages/vtea-core/README.md` for the current
module-by-module status). Phase 4 (napari GUI, Option A) is in progress.
