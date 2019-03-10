"""
Extract and build local DBs with relevant information.

Prerequisites:

- From LIPIDMAPS, file LMSDFAll.sdf, obtained (and renamed) from

http://www.lipidmaps.org/resources/downloads/index.html
LMSD Structure-data file (SDF)-> extract and expand LMSDDownload<date>All.sdf.

this file is used to generate lm_metadata.txt (tab delimited).


- From Human Metabolomics Data Base, file hmdb_metabolites.zip, obtained from
http://www.hmdb.ca/downloads -> All metabolites data set.

this file is used to generate trans_hmdb2kegg.txt.

"""
import requests
from zipfile import ZipFile
import time
import xml.etree.ElementTree as ET
import zipfile
from six import StringIO


def fetch_db(url, file_name):
    """Fetch and extract online data bases' .zip files"""
    start = time.time()
    # download the file contents in binary format
    print('Downloading...')
    s = requests.Session()
    r = s.get(url, stream=True)
    
    # open method to open a file on your system and write the contents
    print('Download complete. Creating the files...')
    with open(file_name, "wb") as f:
        f.write(r.content)

    # opening the zip file in READ mode 
    with ZipFile(file_name, 'r') as zip: 
        # printing all the contents of the zip file 
        zip.printdir() 
  
        # extracting all the files 
        print('Extracting all the files now...') 
        zip.extractall() 
        print('Done!')
    end = time.time()
    print('It took', end-start, 'seconds to fetch this data base.')
    
    
print('Fetching hmbd')
fetch_db('http://www.hmdb.ca/system/downloads/current/hmdb_metabolites.zip', 'hmbd_db.zip')
print('Fetching LIPIDMAPS')
fetch_db('http://www.lipidmaps.org/resources/downloads/LMSDFDownload12Dec17.zip', 'LIPIDMAPS_db.zip')
print('Fetching ChEBI')
fetch_db('ftp://ftp.ebi.ac.uk/pub/databases/chebi/SDF/ChEBI_complete.sdf.gz', 'ChEBI_db.zip')

# Extract information from metadata of SDF files of LIPIDMAPS

def get_sdf_records(lmfilename):
    """Generator to yield one SDF record at a time."""
    with open(lmfilename) as lmfile:
        curr_record = []
        for line in lmfile:
            line = line.strip()
            if line == "$$$$": # record separator
                result = '\n'.join(curr_record)
                result = result.strip()
                if len(result) > 0:
                    yield result
                curr_record = []
            else:
                curr_record.append(line)

def get_lm_metafields(record):
    result = {}
    metas = record.split('M  END', 1)[1]
    split_metas = metas.split('> <')
    for s in split_metas:
        s = s.strip()
        if len(s) == 0:
            continue
        #print(s)
        key, value = s.split('>', 1)
        result[key.strip()] = value.strip()
    return result
        
all_lipids_file = 'metabo_dbs/LMSDFAll.sdf'

out_lm_file = 'dbs/lm_metadata.txt'

fields = ('CATEGORY', 'FORMULA', 'HMDBID', 'CHEBI_ID', 'KEGG_ID',
          'MAIN_CLASS', 'SUB_CLASS', 'SYSTEMATIC_NAME')

print ('processing {} ...'.format(all_lipids_file))

with open(out_lm_file, 'w') as outfile:
    count = 0
    print('LM_ID\t'+'\t'.join(fields), file=outfile)
    
    for record in get_sdf_records(all_lipids_file):
        metadata = get_lm_metafields(record)
        lm_id = metadata['LM_ID']
        print(lm_id)
        line = [lm_id]
        
        for field in fields:
            if field in metadata:
                line.append(metadata[field])
            else:
                line.append('')
        
        print('\t'.join(line), file=outfile)
        count +=1
    
print('{} records processed in {}'.format(count, all_lipids_file))
print('File {} generated'.format(out_lm_file))

# Extract information from HMDB

all_mets_file = 'metabo_dbs/hmdb_metabolites.zip'
print('\n------------------------------------------\n')
print ('Creating conversions HMDB->KeGGId...')

conv = {}

with zipfile.ZipFile(all_mets_file) as zip:
    for name in zip.namelist():
        if name.startswith('hmdb_metabolites'):
            continue
        record = zip.read(name).decode('utf8')
        
        tree = ET.parse(StringIO(record))
        root = tree.getroot()
        
        accession_tag = root.find('accession')
        acc = accession_tag.text.strip()
        
        kegg_tag = root.find('kegg_id')
        if not kegg_tag.text is None:
            kegg_id = kegg_tag.text.strip()
            if not kegg_id.startswith('.'):
                conv[acc] = kegg_id
                print('{} -> {}'.format(acc, kegg_id))

n_mets = len(zip.namelist())
n_trans = len(conv)
print('{} compounds in HMDB. {} have a KeGGId'.format(n_mets, n_trans))

out_filename = 'dbs/trans_hmdb2kegg.txt'

# try to follow KeGG format for Id translation.
line_frmt = 'hmdb:{}\tcpd:{}\tequivalent'.format

with open(out_filename, 'w') as outfile:
    for hmdb in sorted(list(conv.keys())):
        print(line_frmt(hmdb, conv[hmdb]), file=outfile)

print(out_filename, 'was created')

