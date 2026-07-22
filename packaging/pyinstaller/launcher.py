"""PyInstaller entry script - thin wrapper so the spec file has a stable,
predictable script to point Analysis() at regardless of how vtea_napari's
own entry point is packaged/installed."""

from vtea_napari.app import main

if __name__ == "__main__":
    main()
