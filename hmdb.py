import zipfile as zf
import xml.etree.ElementTree as ET
import requests

def hmdb_local (x):
    '''Parse classes of a coumpound using local HMDB database.'''
    
    with zf.ZipFile('hmdb_metabolites.zip') as myzip:
        with myzip.open(x+'.xml') as myfile:
            root = ET.fromstring(myfile.read())
            a = root.find('./taxonomy/kingdom').text
            b = root.find('./taxonomy/super_class').text
            c = root.find('./taxonomy/class').text
    return (a,b,c)

def hmdb_web (x):
    '''Parse classes of a coumpound using the HMDB website.'''
    
    r = requests.get('http://www.hmdb.ca/metabolites/'+x+'.xml')
    root = ET.fromstring(r.text)
    a = root.find('./taxonomy/kingdom').text
    b = root.find('./taxonomy/super_class').text
    c = root.find('./taxonomy/class').text
    return (a,b,c)

