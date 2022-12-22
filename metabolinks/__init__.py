from .dataio import (read_data_from_xcel, read_data_csv)
from .datasets import parse_data, get_data_path
from .peak_alignment import align

# this registers "column-organized" data accessors
from .cdaccessors import (CDLAccessor, CDFAccessor, add_labels)

__version__ = '0.75'
