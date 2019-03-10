""" 
Extract and organize ontology information from .obo files, 
more specifically ChemOnt.

Prerequisites:  ChemOnt_2_1.obo file  

"""
import time

def read_OBO(filename):
    """Parse OBO files and build hierarchy tables.
    
    Returns tables of term names and list of children (as dicts)."""

    with open(filename, encoding='utf_8') as f:
        content = f.read()
    
    #create a list of terms
    terms= content.split('\n\n')
    #remove empty lines
    terms= [t for t in terms if len(t) > 0]
    #remove version info, etc.
    terms.pop(0)

    names_table = {}
    children_table = {}
    
    for t in terms:
        # extract term ID, name and parent from term record
        lines = t.splitlines()
        for line in lines:
            if line.startswith('id:'):
                term_id = line.split('CHEMONTID:')[1]
            if line.startswith('name:'):
                name = line.split(': ')[1]
            if line.startswith('is_a'):
                parent = line.split('CHEMONTID:')[1].split(' ! ')[0]

        # fill tables

        names_table[term_id] = name

        if parent in children_table:
            children_table[parent].append(term_id)
        else:
            children_table[parent] = [term_id]

    return names_table, children_table

def show_hierarchy(names_table, children_table):
    for k, kingdom in enumerate(children_table['9999999']):
        print(k + 1, names_table[kingdom])
        for s, superclass in enumerate(children_table.get(kingdom, ())):
            print(f'   {k+1}.{s} {names_table[superclass]}')
            for c, cclass in enumerate(children_table.get(superclass, ())):
                print(f'      {k+1}.{s}.{c} {names_table[cclass]}')
                for b, sclass in enumerate(children_table.get(cclass, ())):
                    print(f'         {k+1}.{s}.{c}.{b} {names_table[sclass]}')


if __name__ == '__main__':
    print('started!')
    start_time = time.time()

    #read ChemOnt_2_1.obo file
    names_table, children_table = read_OBO('ChemOnt_2_1.obo')

    elapsed = time.time() - start_time

    print(f'Tables created. Elapsed time = {elapsed:.3f} s\n')

    ## for i, t in zip(range(20), children_table):
    ##     print(t, ':', names_table[t], '-->', children_table[t])

    show_hierarchy(names_table, children_table)
