from enum import Enum

class RowCorruptionTypes(Enum):
    SWAP_ROWS = "swap_rows"
    DELETE_ROWS = "delete_rows"
    SHUFFLE_COLUMNS = "shuffle_columns"
    REVERSE_ROWS = "reverse_rows"

class CellCorruptionTypes(Enum):
    OUTLIER = "outlier"
    NULL = "null"
    INCORRECT_DATATYPE = "incorrect_datatype"
    INCONSISTENT_FORMAT = "inconsistent_format"  # TODO: Implement
    SWAP_CELLS = "swap_cells"
    CASE_ERROR = "case_error"
    TRUNCATE = "truncate"
    ROUNDING_ERROR = "rounding_error"
    ENCODING_ERROR = "encoding_error"  # TODO: Implement
    TYPO = "typo"
