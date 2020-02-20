import pandas as pd

from metabolinks.dataio import read_MassTRIX
from metabolinks.transformations import unfold_MassTRIX
from metabolinks.elementcomp import element_composition


def cleanup_cols(df, isotopes=True, uniqueID=True, columns=None):
    """Removes the 'uniqueID' and the 'isotope presence' columns."""
    col_names = []
    if uniqueID:
        col_names.append("uniqueID")
    if isotopes:
        iso_names = (
            "C13",
            "O18",
            "N15",
            "S34",
            "Mg25",
            "Mg26",
            "Fe54",
            "Fe57",
            "Ca44",
            "Cl37",
            "K41",
        )
        col_names.extend(iso_names)
    if columns is not None:
        col_names.extend(columns)
    return df.drop(col_names, axis=1)


file_name = "MassTRIX_output.tsv"
import os

_THIS_DIR, _ = os.path.split(os.path.abspath(__file__))
testfile_name = os.path.join(_THIS_DIR, "data", file_name)

results = read_MassTRIX(testfile_name)
results = cleanup_cols(results)

print(f"File {testfile_name} was read\n")

results.info()

print("\n+++++++++++++++++++++++++++++++")
print("Element compositions:")

compositions = ["CHO", "CHOS", "CHON", "CHONS", "CHOP", "CHONP", "CHONSP"]

elem_comp = element_composition(
    results, column="KEGG_formula", compositions=compositions
)

for c in compositions + ["other"]:
    print(c, elem_comp[c])

print("\n+++++++++++++++++++++++++++++++")

unfolded = unfold_MassTRIX(results)

print("Unfolded dataframe:\n")
print("is of type", type(unfolded))
unfolded.info()

print("---------------------")
print(unfolded.head(10))
