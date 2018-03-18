def parse4(comps, *args):
    data = []
    for comp in comps:
        l = []
        for arg in args:
            if arg not in comp:
                l.append('None')
            else:
                metadata = comp.split('> ')
                for item in metadata:
                    if arg in item:
                        l.append(''.join(item.split('\n')[1:2]).replace('CHEBI:', ''))
        data.append(l)
    return data
# Function: searches for arguments metadata within each compounds data.
# Return: a set of lists, each containing 1 compound's arguments metadata.
# Input: comps=compounds list; args=arguments/parameters

# ChEBI Parsing.
file_name = 'sample_input_chebi.sdf'
with open(file_name) as f:
    compounds = f.read().split('$$$$')[:-1]
arguments = ['<ChEBI Name>',
             '<IUPAC Names>',
             '<Formulae>',
             '<Monoisotopic Mass>',
             '<ChEBI ID>',
             '<KEGG COMPOUND Database Links>',
             '<HMDB Database Links>',
             '<LIPID MAPS instance Database Links>',
             ]
mass_index = 3  # Identify Mass arg position.
lipidmaps_index = 7  # Identify Lipid Maps ID arg position.
chebi_data = parse4(compounds, *arguments)
chebi_data.sort(key=lambda x: x[mass_index])

with open('sample_output_chebi.tsv', 'w') as f:
    # Print header.
    print('\t'.join(arguments).replace('<', '').replace('>', ''), file=f)
    for compound in chebi_data:
        print('\t'.join(compound), file=f)

file_name = 'sample_input_lipidmaps.sdf'
with open(file_name) as f:
    compounds = f.read().split('$$$$')[:-1]
# Argumen positions must be the same for ChEBI and Lipid Maps.
arguments = ['<COMMON_NAME>',
             '<SYSTEMATIC_NAME>',
             '<FORMULA>',
             '<EXACT_MASS>',
             '<CHEBI_ID>',
             '<KEGG_ID>',
             '<HMDBID>',
             '<LM_ID>',
             ]
mass_index = 3  # Identify Mass arg position.
lipidmaps_index = 7  # Identify Lipid Maps ID arg position.
lipidmaps_data = parse4(compounds, *arguments)
lipidmaps_data.sort(key=lambda x: x[mass_index])

with open('sample_output_lipidmaps.tsv', 'w') as f:
    # Print header.
    print('\t'.join(arguments).replace('<', '').replace('>', ''), file=f)
    for compound in lipidmaps_data:
        print('\t'.join(compound), file=f)

# Joining the two data sets. Lipid Maps data prioritized.
mash_up_data = lipidmaps_data
temp = [x[lipidmaps_index] for x in lipidmaps_data if x != 'None']
for comp in chebi_data:
    if comp[lipidmaps_index] == 'None':
        mash_up_data.append(comp)
    elif comp[lipidmaps_index] != 'None' and comp[lipidmaps_index] not in temp:
        mash_up_data.append(comp)
        temp.append(comp[7])

# The next segment will:
# - Convert order_by_index data into float variables.
# - Order data.
# - Convert back into strings for 'join' method.
mash_up_sort = []
for item in mash_up_data:
    if item[mass_index] == 'None':
        item[mass_index] = float(999999.999999)
        mash_up_sort.append(item)
    else:
        item[mass_index] = float(item[mass_index])
        mash_up_sort.append(item)
mash_up_sort.sort(key=lambda x: x[mass_index])
mash_up_data = [map(str, x) for x in mash_up_sort]

with open('sample_output_mash_up.tsv', 'w') as f:
    # Print header.
    print("Common Name",
          "Systematic Name",
          "Formula",
          "Monoisotopic Mass",
          "ChEBI ID",
          "KEGG Compound ID",
          "HMBD ID",
          "Lipid Maps ID", sep='\t', file=f)
    for item in mash_up_data:
        print('\t'.join(item), file=f)
