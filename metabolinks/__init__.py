from .dataio import (read_data_from_xcel, read_data_csv, parse_data)
from .datasets import get_data_path
from .peak_alignment import align

# this registers "column-organized" data accessors
from .cdaccessors import (CDLAccessor, CDFAccessor, add_labels)

__version__ = '0.79'

__all__ = ["read_data_from_xcel",
           "read_data_csv",
           "parse_data",
           "get_data_path",
           "CDLAccessor",
           "CDFAccessor",
           "add_labels",
           "align",
           "__version__",]
