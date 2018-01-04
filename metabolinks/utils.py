"""utility functions:

   - tests for common types
   - simple format of h:m:s from seconds
"""

from __future__ import print_function, absolute_import
from six import string_types, integer_types

import tkinter as tk

def get_filename_using_tk():
    """Choose a filename using Tk"""
    root = tk.Tk()
    root.withdraw()
    fname = tk.filedialog.askopenfilename(filetypes = [("TSV","*.tsv")])
    print('Selected file {}'.format(fname))
    return fname


def _is_sequence(arg):
    isstring = isinstance(arg, string_types)
    isothersequences = hasattr(arg, "__getitem__") or hasattr(arg, "__iter__")
    return not isstring and isothersequences


def _is_string(a):
    return isinstance(a, string_types)


def _is_number(a):
    return isinstance(a, float) or isinstance(a, integer_types)


def s2HMS(seconds):
    m, s = divmod(seconds, 60.0)
    h, m = divmod(m, 60.0)
    if h == 0:
        return "%02dm %06.3fs" % (m, s)
    return "%dh %02dm %06.3fs" % (h, m, s)
