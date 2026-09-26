"""Writing results in formats that outlive VTEA - see
docs/SAVING_AND_ARCHIVING.md. The feature table and its data dictionary
first; figures and the archive bundle build on this."""

from vtea_core.export.table import (
    DICTIONARY_SUFFIX,
    TABLE_FORMATS,
    data_dictionary,
    dictionary_path_for,
    export_table,
)

__all__ = [
    "DICTIONARY_SUFFIX",
    "TABLE_FORMATS",
    "data_dictionary",
    "dictionary_path_for",
    "export_table",
]
