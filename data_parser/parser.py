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
                        l.append(','.join(item.split('\n')[1:-2]))
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
order_by_index = 2  # Identify Monoisotopic Mass arg position.
chebi_data = parse4(compounds, *arguments)
chebi_data.sort(key=lambda x: x[order_by_index])

with open('sample_output_chebi.tsv', 'w') as f:
    # Print header.
    print('\t'.join(arguments).replace('<', '').replace('>', ''), file=f)
    for compound in chebi_data:
        print('\t'.join(compound), file=f)

file_name = 'sample_input_lipidmaps.sdf'
with open(file_name) as f:
    compounds = f.read().split('$$$$')[:-1]
arguments = ['<COMMON_NAME>',
             '<SYSTEMATIC_NAME>',
             '<FORMULA>',
             '<EXACT_MASS>',
             '<CHEBI_ID>',
             '<KEGG_ID>',
             '<HMDBID>',
             '<LM_ID>',
             ]
order_by_index = 2  # Identify Mass arg position.
lipidmaps_data = parse4(compounds, *arguments)
lipidmaps_data.sort(key=lambda x: x[order_by_index])

with open('sample_output_lipidmaps.tsv', 'w') as f:
    # Print header.
    print('\t'.join(arguments).replace('<', '').replace('>', ''), file=f)
    for compound in lipidmaps_data:
        print('\t'.join(compound), file=f)