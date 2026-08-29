# Saving and archiving VTEA work

Design for making a VTEA session reloadable, and for exporting it in a form
that can be deposited with a paper. Two pieces are already built - the gate
JSON (`vtea_core.gates.io`) and the feature catalog behind the data
dictionary (`vtea_core.measurements.FeatureCatalog`); this document is the
plan for the rest.

## The problem

Right now a session lives entirely in the widget. Close napari and the
protocol, the gates and the measurement table are gone. Three different
people need three different things out of it:

| Who | Wants | Size |
| --- | --- | --- |
| The same person tomorrow | The protocol back, to re-run on the next image | kilobytes |
| A colleague debugging a result | Everything, including the intermediate images | gigabytes |
| A reader of the paper | The numbers and figures behind Figure 3, readable without VTEA | megabytes |

One format can't serve all three, so there are three, layered: each tier is
the previous one plus more.

## Tier 1 — the protocol (`*.vtea.json`)

Everything needed to re-run the analysis, and nothing that can be
recomputed. This is the file people will actually share.

```json
{
  "vtea_protocol_version": 1,
  "created": "2026-08-29T10:15:00Z",
  "source": {
    "path": "sample_042.ome.tif",
    "sha256": "9f2b…",
    "shape": [24, 3, 2048, 2048],
    "channel_axis": 1,
    "z_axis": 0
  },
  "processing": [
    {
      "name": "threshold_mask_1",
      "category": "segmentation",
      "function": "threshold_mask",
      "params": {"method": "otsu"},
      "input_keys": {"volume": "volume"},
      "output_key": "mask",
      "channel": 0
    }
  ],
  "analysis": [
    {
      "name": "extract_measurements_by_channel_1",
      "category": "measurements",
      "function": "extract_measurements_by_channel",
      "params": {},
      "input_keys": {"labels": "watershed_split_1", "intensity": "intensity"},
      "output_key": "measurements",
      "channel": null,
      "features": []
    },
    {
      "name": "kmeans_1",
      "category": "clustering",
      "function": "kmeans",
      "params": {"n_clusters": 4},
      "input_keys": {"data": "data"},
      "output_key": "clusters",
      "channel": null,
      "features": ["mean_ch0", "mean_ch2"]
    }
  ],
  "gates": { "...": "the vtea_core.gates.io payload, inlined" },
  "features": { "...": "the vtea_core.measurements.FeatureCatalog payload" },
  "plot": {"x": "mean_ch0", "y": "kmeans_1", "color_by": "count", "colormap": "viridis"},
  "environment": {
    "vtea_core": "0.1.1",
    "vtea_napari": "0.1.1",
    "python": "3.11.9",
    "packages": {"scikit-image": "0.24.0", "scikit-learn": "1.5.1", "numpy": "2.0.1"}
  }
}
```

Notes on the shape of this:

- **The `Step` dataclass already is this record.** Every field above exists
  on it; the work is `to_dict`/`from_dict` plus a version check, mirroring
  `vtea_core.gates.io` exactly. Put it in `vtea_core/workflow/io.py`.
- **`features` is why a clustering is reproducible at all.** An empty list
  means "every measured feature", which is deliberately not expanded on
  save: a protocol re-run against an image with four channels rather than
  three should use the features that image has, not the ones the original
  had. A non-empty list is the explicit choice, recorded verbatim.
- **`name` is what makes the file readable and stable.** `input_keys`
  referring to `watershed_split_1` says which segmentation was measured; a
  file that only recorded `"labels"` would be ambiguous the moment a
  protocol has two of them.
- **The source image is identified, not embedded.** A path plus a SHA-256 is
  enough to check you re-ran against the same data, and keeps the file
  small enough to email or commit.
- **`environment` is the difference between "documented" and
  "reproducible".** `threshold_otsu` changing between scikit-image releases
  moves every downstream number; without the versions there is no way to
  tell that happened.
- **Non-JSON parameters need a rule.** A trained classifier or a numpy array
  passed as a parameter can't be inlined. Write those to a sidecar file and
  store a reference (`{"$file": "models/classifier_1.joblib"}`), which
  degrades honestly: the protocol still opens, the step reports the missing
  file.

## Tier 2 — the session archive (`*.vtea.zip`)

Tier 1 plus everything the run produced, so a colleague can look at the
intermediate images without re-running a two-hour segmentation.

```
session.vtea.zip
├── manifest.json          # every file below, with sha256 and byte size
├── protocol.vtea.json     # tier 1, verbatim
├── gates.json
├── data/
│   ├── measurements.csv         # the flat feature table
│   └── data_dictionary.csv      # one row per column - see tier 3
├── derived/
│   ├── threshold_mask_1.ome.zarr/    # named after the step, not "step2"
│   ├── watershed_split_1.ome.zarr/
│   └── ...
├── figures/
│   ├── scatter_mean_ch0_vs_kmeans_1.svg
│   └── scatter_mean_ch0_vs_kmeans_1.png
└── README.md              # generated: what this is, how to re-open it
```

- **Zarr, not TIFF, for intermediates.** They are chunked and compressed,
  `vtea_core.io.zarr_io` already reads and writes them, and a label volume
  compresses to a fraction of its size. OME-Zarr metadata makes them open in
  napari, Fiji or a browser viewer without VTEA.
- **A zip, not a directory, by default.** One file to move, and the release
  work already established that thousands of small files are painful to
  unpack — Zarr chunk directories are exactly that shape. Store the zip
  uncompressed (the chunks are already compressed) so it stays streamable.
- **`manifest.json` carries a checksum per file.** That is what makes a
  bundle verifiable years later, and it is what a repository like Zenodo
  will show alongside the deposit.
- **Intermediates are opt-in per step.** A checkbox on each step card
  ("archive this result"); default on for segmentations, off for
  preprocessing, which is both large and cheap to recompute.

## Tier 3 — the publication bundle

Tier 2 minus the bulk, plus the metadata a repository needs. This is what
gets a DOI.

Everything is in a format that outlives VTEA:

| Content | Format | Why |
| --- | --- | --- |
| Feature table | CSV (UTF-8, one row per object) | Opens in anything; no VTEA needed |
| Column meanings | `data_dictionary.csv` | See below |
| Gates | JSON | Already shipped |
| Protocol | JSON | Tier 1 |
| Figures | SVG **and** PNG | SVG is vector, which journals ask for; PNG for preview |
| Deposit metadata | `datacite.json` + `CITATION.cff` | What Zenodo/Dryad read to mint a DOI |
| Licence | `LICENSE` | Reuse is not possible without one |

**The data dictionary is the part that is easy to skip and shouldn't be.** A
column called `mean_ch2` is meaningless to a reader; one row of a dictionary
makes it self-describing:

| column | kind | measurement | channel | segmentation | produced_by | params | source_features | units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mean_ch2` | intensity | mean | 2 | `watershed_split_1` | `extract_measurements_by_channel_1` | | | a.u. |
| `count` | geometry | count | | `watershed_split_1` | `extract_measurements_by_channel_1` | | | voxels |
| `kmeans_1` | derived | cluster assignment | | `watershed_split_1` | `kmeans_1` | `n_clusters=4` | `mean_ch0, mean_ch2` | |

**This part is built** (`vtea_core.measurements.FeatureCatalog`). Every cell
is recorded as the step that produces the column runs, rather than guessed at
afterwards from the column name: the channel comes from the measurement
step's naming, the segmentation from its `input_keys["labels"]`,
`produced_by` from the step name, and `source_features` from the feature
selection the clustering or reduction step was given. `catalog.to_dataframe()`
is the dictionary above; `catalog.to_dict()` is its JSON form, ready to write
into the bundle. What remains is the writing.

`source_features` is the row that earns the table: without it, "these cells
were clustered" is an assertion, and with it a reader can see that the
clustering used two intensity features and not the object sizes.

## How this maps to FAIR

- **Findable** — `datacite.json` gives the deposit a title, authors,
  keywords and a DOI; `manifest.json` gives every file inside it a checksum
  to cite precisely.
- **Accessible** — open formats only, no pickles: JSON, CSV, OME-Zarr,
  SVG/PNG. Nothing in the bundle requires VTEA, or Python, to read.
- **Interoperable** — OME-Zarr and OME-TIFF are the community standards for
  bioimaging; the feature table is tidy CSV with a dictionary; the protocol
  gets a published JSON Schema so other tools can read or generate one.
- **Reusable** — provenance (package versions, parameters, source-image
  checksum) plus an explicit licence, and the protocol re-runs, so a result
  can be checked rather than taken on trust.

The single biggest reproducibility risk is the source image: a bundle that
identifies it only by filename is not reproducible if that file is on
someone's laptop. The export dialog should say so plainly and offer to
include the source (opt-in, because it is usually the largest thing in the
bundle) or to record a DOI/accession for it where it is already deposited.

## Build order

Each step is independently useful, so this can stop at any point and still
have delivered something.

1. **`vtea_core/workflow/io.py`** — `pipeline_to_dict`/`pipeline_from_dict`,
   `save_protocol`/`load_protocol`, versioned, mirroring `gates/io.py`.
   Round-trip tests: a loaded protocol re-run on the same input produces the
   same table.
2. **Save/Open protocol in the builder** — two buttons, plus "include gates".
   This alone removes the "close napari and lose everything" problem.
3. **`environment` capture** — `importlib.metadata` over a short list of
   packages whose behaviour affects results.
4. **CSV + data dictionary export** — `vtea_core/export/table.py`. The
   dictionary itself is already assembled at run time by `FeatureCatalog`,
   so this step is `to_dataframe().to_csv()` plus the table beside it.
5. **Figure export** — the plot already has a matplotlib `Figure`;
   `savefig` to SVG and PNG, named for the axes.
6. **`vtea_core/export/bundle.py`** — the manifest, and the zip writer.
7. **Session archive** — intermediates to Zarr with the per-step opt-in.
8. **`datacite.json` / `CITATION.cff`** — a small metadata form in the
   export dialog (title, authors, ORCIDs, licence, keywords).

Steps 1–3 are the ones worth doing next; 4–5 make the first publishable
export; 6–8 complete the archive.
