"""PyInstaller entry script - thin wrapper so the spec file has a stable,
predictable script to point Analysis() at regardless of how vtea_napari's
own entry point is packaged/installed."""

import sys

from vtea_napari.app import main

if __name__ == "__main__":
    # sys.exit() matters here: main() returns a non-zero code for a failed
    # --self-test, and without propagating it the packaged binary would
    # report success to CI no matter what the self-test found.
    sys.exit(main())
