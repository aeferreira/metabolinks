""" 
Extract an organize ontology information from .obo files, 
more specifically ChemOnt.
Prerequisites:  ChemOnt_2_1.obo file  
"""

def extract_term(lines):
    #extract terms from lines
    for l in lines:
        if l.startswith('name:'):
            parts_list= l.split(': ')
            name= parts_list[1]
            return name

def extract_parent_class(lines):
    #extract parent class from lines
    for l in lines:
        if l.startswith('is_a'):
            parts_list= l.split('! ')
            parent= parts_list[1]
            return parent

def list_iter(list_of_lists):
    #iteration of superclass terms
    for j in range(len(list_of_lists)):
        for i in list_of_lists[j]:
            yield i


#read ChemOnt_2_1.obo file
with open('ChemOnt_2_1.obo', encoding='utf_8') as f:
    file= f.read()

#create a list of terms
term_list= file.split('\n\n')
#remove empty lines
term_list= [t for t in term_list if len(t) > 0]
#remove version info, etc.
term_list.pop(0)

#create lists for each hierarchical class
kingdom=['Organic compounds', 'Inorganic compounds']
superclass_temp=[]
superclass_complete=[]
cclass_temp=[]
cclass_complete=[]
subclass_temp=[]
subclass_complete=[]

for k in kingdom:
    superclass_temp.clear()
    for t in term_list:
        #extract terms and parent class from lines
        lines= t.splitlines()
        term= extract_term(lines)
        parent_class= extract_parent_class(lines)

        #organize superclasses by kingdom
        if parent_class== k:
            superclass_temp.append(term)
    superclass_complete=superclass_complete+[superclass_temp.copy()]

for i in list_iter(superclass_complete):
    cclass_temp.clear()
    for t in term_list:
         #extract terms and parent class from lines
        lines= t.splitlines()
        term= extract_term(lines)
        parent_class= extract_parent_class(lines)

        #organize classes by superclasses
        if parent_class== i:
            cclass_temp.append(term)
    cclass_complete=cclass_complete+[cclass_temp.copy()]

for i in list_iter(cclass_complete):
    subclass_temp.clear()
    for t in term_list:
         #extract terms and parent class from lines
        lines= t.splitlines()
        term= extract_term(lines)
        parent_class= extract_parent_class(lines)

        #organize subclasses by classes
        if parent_class== i:
            subclass_temp.append(term)
    subclass_complete=subclass_complete+[subclass_temp.copy()]

#show hierarchy
for a,b in enumerate(kingdom):
    print(a+1, b, '\n')
    for c in range(len(superclass_complete[a])):
        print(f' {a+1}.{c} {superclass_complete[a][c]} \n')
        for d in range(len(cclass_complete[c])):
            print(f'    {a+1}.{c}.{d} {cclass_complete[c][d]} \n' )
            for e in range(len(subclass_complete[d])):
                print(f'         {a+1}.{c}.{d}.{e} {subclass_complete[d][e]} \n')