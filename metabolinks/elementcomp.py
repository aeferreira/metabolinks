from collections import Counter

def element_composition(df, column=None,
                            compositions = ('CHO', 'CHOS',
                                                'CHON', 'CHONS',
                                                'CHOP', 'CHONP',
                                                'CHONSP'),
                            verbose=True):
    
    formulae = df[column].str.split('#').apply(set).apply(list)
    # print(formulas)

    # remove unambigous formulae
    is_1 = [True if len(f) == 1 else False for f in formulae.values]
    formulae = formulae[is_1]
    
    #print(formulae)
    
    # convert to list of strings and remove duplicates
    formulae = list(set([f[0] for f in formulae.values]))
    #print(formulae)
    if verbose:
        print(len(formulae), 'formulae')

    # Calculate element compositions
    comps = []
    for formula in formulae:
        # remove numbers
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
    
##     for f, c in zip(formulae, comps):
##         print(f, c)
        
    return elem_comp

if __name__ == '__main__':

    from metabolinks.masstrix import read_MassTRIX
    testfile_name = '../example_data/masses.annotated.reformat.tsv'

    df = read_MassTRIX(testfile_name).cleanup_cols()
    
    print("File {} was read\n".format(testfile_name))
    
    df.info()

    print('\n+++++++++++++++++++++++++++++++')
    compositions = ['CHO', 'CHOS', 'CHON', 'CHONS', 
                    'CHOP', 'CHONP', 'CHONSP']
    
    elem_comp = element_composition(df, column='KEGG_formula',
                                    compositions=compositions)
    for c in compositions + ['other']:
        print(c, elem_comp[c])

