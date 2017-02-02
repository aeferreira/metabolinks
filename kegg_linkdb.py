import requests

def linkdb(db):

	'''Get the evivalencies between the KEGG COMPOUND database and a database of choice.'''
    
    payload={'page':'download',
             'e':'equiv',
             'm':'compound',
             't':db,
             'targetformat':None}
    
    possible_db = ["pubchem","chebi","pdb-ccd","3dmet","lipidmaps","lipidbank","knapsack",
                   "hmdb", "hsdb","massbank","nikkaji","chembl"]
    
    if db not in possible_db:
        print (db+': not found')
        print ('Please use one of the folowing:')
        print (possible_db)
        return
    
    r = requests.post('http://www.genome.jp/dbget-bin/get_linkdb_list', data=payload)
    
    return bytes.decode(r.content)


