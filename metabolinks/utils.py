from __future__ import print_function

import tkinter as tk

def get_filename_using_tk():
    """Choose a filename using Tk"""
    root = tk.Tk()
    root.withdraw()
    fname = tk.filedialog.askopenfilename(filetypes = [("TSV","*.tsv")])
    print('Selected file {}'.format(fname))
    return fname