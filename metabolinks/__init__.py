# this registers "column-organized" data accessors
from .cdaccessors import CDFAccessor, CDLAccessor, add_labels
from .dataio import parse_data, read_data_csv, read_data_from_xcel
from .datasets import get_data_path
from .peak_alignment import align

__version__ = '0.80'

__all__ = [
           "CDFAccessor",
           "CDLAccessor",
           "__version__",
           "add_labels",
           "align",
           "get_data_path",
           "parse_data",
           "read_data_csv",
           "read_data_from_xcel",
]

print("metabolinks was imported!!!")
print("version:", __version__)