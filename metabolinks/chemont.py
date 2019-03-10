""" 
Extract and organize ontology information from .obo files, 
more specifically ChemOnt.

Prerequisites:  ChemOnt_2_1.obo file  

"""
import time

def extract_name(lines):
    #extract terms from lines
    for l in lines:
        if l.startswith('name:'):
            return l.split(': ')[1]


def extract_parent_class(lines):
    #extract parent class from lines
    for l in lines:
        if l.startswith('is_a'):
            return l.split('! ')[1]


def extract_name_parent(t):
    #extract name and parent class from term record
    name = parent = ""
    lines= t.splitlines()
    for l in lines:
        if l.startswith('is_a'):
            parent= l.split('! ')[1]
        if l.startswith('name:'):
            name =  l.split(': ')[1]
    return (name, parent)


def list_iter(list_of_lists):
    #iteration of 2-level nested lists
    for elem in list_of_lists:
        for i in elem:
            yield i


def read_name_parent_from_OBO(filename):

    with open(filename, encoding='utf_8') as f:
        content = f.read()

    #create a list of terms
    terms= content.split('\n\n')
    #remove empty lines
    terms= [t for t in terms if len(t) > 0]
    #remove version info, etc.
    terms.pop(0)

    term_table = []
    for t in terms:
        # extract name and parent class from each term
        # create list of (term, parent) tuples
        lines = t.splitlines()
        name = extract_name(lines)
        parent_class = extract_parent_class(lines)
        term_table.append((name, parent_class))
    return term_table

print('started!')
start_time = time.time()

#read ChemOnt_2_1.obo file
term_table = read_name_parent_from_OBO('ChemOnt_2_1.obo')

kingdoms = ['Organic compounds', 'Inorganic compounds']

superclasses = []
for k in kingdoms:
    templist = []
    for term, parent_class in term_table:
        #organize superclasses by kingdom
        if parent_class == k:
            templist.append(term)
    superclasses.append(templist)

## for i in superclasses:
##     print(i)

classes = []
for k in list_iter(superclasses):
    templist = []
    for term, parent_class in term_table:
        #organize classes by superclasses
        if parent_class == k:
            templist.append(term)
    classes.append(templist)

## inorganic_offset = len(superclasses[0])
## for i in classes[inorganic_offset]:
##     print(i)

subclasses = []
for k in list_iter(classes):
    templist = []
    for term, parent_class in term_table:
        #organize subclasses by classes
        if parent_class == k:
            templist.append(term)
    subclasses.append(templist)

elapsed = time.time() - start_time
print(f'Done. Elapsed time = {elapsed:.3f} s\n')

#show hierarchy
c_index = 0
sc_index = 0
for k, kname in enumerate(kingdoms):
    print(k + 1, kname)

    for s, superclass_name in enumerate(superclasses[k]):
        print(f' {k + 1}.{s} {superclass_name}')

        for c, class_name in enumerate(classes[c_index]):
            print(f'    {k + 1}.{s}.{c} {class_name}' )

            for b, subclass_name in enumerate(subclasses[sc_index]):
                print(f'         {k + 1}.{s}.{c}.{b} {subclass_name}')
            sc_index += 1
        
        c_index += 1

