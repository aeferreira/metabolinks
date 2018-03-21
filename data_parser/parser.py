def parse4(compounds, *args):
    data = []
    for c in compounds:
        l = []
        for arg in args:
            if arg not in c:
                l.append('None')
            else:
                metadata = c.split('> ')
                for i in metadata:
                    if arg in i:
                        l.append(''.join(i.split('\n')[1:2]).replace('CHEBI:', ''))
        data.append(l)
    return data
# Function: searches for arguments metadata within each compounds data.
# Return: a set of lists, each containing 1 compound's arguments metadata.
# Input: compounds=compounds list; args=arguments/parameters

# ChEBI Parsing.
file_name = 'sample_input_chebi.sdf'
with open(file_name) as f:
    compounds = f.read().split('$$$$')[:-1]
args = ['<ChEBI Name>',
        '<IUPAC Names>',
        '<Formulae>',
        '<Monoisotopic Mass>',
        '<ChEBI ID>',
        '<KEGG COMPOUND Database Links>',
        '<HMDB Database Links>',
        '<LIPID MAPS instance Database Links>'
        ]
chebi_data = parse4(compounds, *args)
with open('sample_output_chebi.tsv', 'w') as f:
    # Print header.
    print("Common Name",
          "Systematic Name",
          "Formula",
          "Monoisotopic Mass",
          "ChEBI ID",
          "KEGG Compound ID",
          "HMBD ID",
          "Lipid Maps ID", sep='\t', file=f)
    for compound in chebi_data:
        print('\t'.join(compound), file=f)

# Lipid Maps Parsing.
file_name = 'sample_input_lipidmaps.sdf'
with open(file_name) as f:
    compounds = f.read().split('$$$$')[:-1]
# Equivalent arg positions must be the same for ChEBI and Lipid Maps.
args = ['<COMMON_NAME>',
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
lipidmaps_data = parse4(compounds, *args)
with open('sample_output_lipidmaps.tsv', 'w') as f:
    # Print header.
    print("Common Name",
          "Systematic Name",
          "Formula",
          "Monoisotopic Mass",
          "ChEBI ID",
          "KEGG Compound ID",
          "HMBD ID",
          "Lipid Maps ID", sep='\t', file=f)
    for lipid in lipidmaps_data:
        print('\t'.join(lipid), file=f)

# Joining the two data sets. Lipid Maps data prioritized.
mash_up_data = lipidmaps_data
temp = [x[lipidmaps_index] for x in lipidmaps_data if x != 'None']
for c in chebi_data:
    if c[lipidmaps_index] == 'None':
        mash_up_data.append(c)
    elif c[lipidmaps_index] != 'None' and c[lipidmaps_index] not in temp:
        mash_up_data.append(c)
        temp.append(c[lipidmaps_index])

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

with open('sample_final_output.tsv', 'w') as f:
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
