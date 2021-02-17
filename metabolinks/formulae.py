from collections import Counter, OrderedDict

import re
import pandas as pd

elem_pattern = r'[A-Z][a-z]?\d*'
elem_groups = r'([A-Z][a-z]?)(\d*)'

def element_composition(formula, elements=None):
    """Given a string with a formula, return dictionary of element composition."""

    composition = {}
    for elemp in re.findall(elem_pattern, formula):
        match = re.match(elem_groups, elemp)
        n = match.group(2)
        number = int(n) if n != '' else 1
        composition[match.group(1)] = number

    if elements is None:
        return composition

    return {e : composition.get(e, 0) for e in elements}

def insert_element_counts(df, formula_column='formula'):
    """Given a pandas DataFrame with a column with formulae, 
       insert new columns with element counts."""

    # Split formulae by elements and store in a new Pandas DataFrame
    formulae = df[formula_column].dropna()
    splits = [element_composition(formula) for formula in formulae.values]
    split_table = pd.DataFrame(splits, index=formulae.index)

    # Insert new columns just after formula column
    formula_index = list(df.columns).index(formula_column) + 1
    for c in reversed(list(split_table.columns)):
        df.insert(formula_index, c, split_table[c])
    return df.copy()

def extract_formulae_from_MassTRIX_records(df, column='KEGG_formula'):
    formulae = df[column].str.split('#').apply(set).apply(list)

    # remove unambigous formulae or empty
    is_1 = [True if len(f) == 1 else False for f in formulae.values]
    formulae = formulae[is_1]
    return [f[0] for f in formulae.values]


def compute_composition_series(formulae, compositions = ('CHO', 'CHOS',
                                                'CHON', 'CHONS',
                                                'CHOP', 'CHONP',
                                                'CHONSP')):

    # remove duplicates
    formulae = pd.unique(formulae)
    # Calculate element compositions
    comps = []
    for formula in formulae:
        # remove numbers and punctuation
        exclude = "0123456789,.[]() "
        for chr in exclude:
            formula = formula.replace(chr,'')
        
        # count according to composition groups
        for c in compositions:
            if set(formula) == set(c):
                comps.append(c)
                break
        else:
            comps.append('other')
    elem_comp = Counter(comps)
    final_comps = OrderedDict()
    labels = list(compositions) + ['other']
    for k in labels:
        final_comps[k] = elem_comp[k]
    return final_comps

def composition_series(formulae, compositions = ('CHO', 'CHOS',
                                                'CHON', 'CHONS',
                                                'CHOP', 'CHONP',
                                                'CHONSP')):

    # Calculate element compositions
    comps = []
    for formula in formulae:
        # remove numbers and punctuation
        exclude = "0123456789,.[]() "
        for chr in exclude:
            formula = formula.replace(chr,'')
        
        # count according to composition groups
        for c in compositions:
            if set(formula) == set(c):
                comps.append(c)
                break
        else:
            comps.append('other')
    return comps

if __name__ == '__main__':
    from io import StringIO
    from metabolinks import datasets
    from metabolinks.dataio import read_MassTRIX

    print('------ test element_composition() ------')
    for test in 'C11H24NO7P', 'C13H19ClN2O2', 'C12H21O11R':
        print(test, '->', element_composition(test))

    print('\n------ test insert_element_counts() ------')
    df = datasets.demo_dataset('table_with_formulae').data
    print(df)
    print('+++++ after insertion ++++++')
    dfi = insert_element_counts(df)
    print(dfi)

    print('\n------ test element_composition_series ------')
    # file_name = "MassTRIX_output.tsv"
    # import os
    # _THIS_DIR, _ = os.path.split(os.path.abspath(__file__))
    # testfile_name = os.path.join(_THIS_DIR, "data", file_name)

    df = read_MassTRIX(StringIO(datasets.create_demo('masstrix_output').as_str()))
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
    keep_cols = ['raw_mass', 'corrected_mass', 'npossible', 'KEGG_mass', 'ppm', 'KEGG_cid', 'KEGG_formula', 'KEGG_name']
    # df = cleanup_cols(df)
    df = df[keep_cols]

    print('\n-------------------------------')
    print(df)
    print('\n---element compositions --------------')
    compositions = ['CHO', 'CHOS', 'CHON', 'CHONS', 
                    'CHOP', 'CHONP', 'CHONSP']
    
    formulae = extract_formulae_from_MassTRIX_records(df, column='KEGG_formula')
    elem_counts = [element_composition(f) for f in formulae]
    elem_counts = pd.DataFrame(elem_counts, index=formulae).fillna(0).astype(int)
    assignments = pd.Series(composition_series(formulae), index=formulae, name='series')
    series_assignments = pd.concat([elem_counts, assignments], axis=1)
    print(series_assignments)
    print('\n-------')
    elem_comp = compute_composition_series(formulae, compositions=compositions)
    for c in elem_comp:
        print(c, elem_comp[c])

    print('\n-------------------------------')
    from metabolinks.transformations import unfold_MassTRIX

    unfolded = unfold_MassTRIX(df).set_index('KEGG_cid')

    print("Unfolded dataframe:\n")
    print("is of type", type(unfolded))
    unfolded.info()
    print("---------------------")
    print(unfolded.head(10))
