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

import xml.etree.ElementTree as ET
import zipfile
from six import StringIO

# Extract information from metadata of SDF files of LIPIDMAPS

def get_lm_records(lmfilename):
    """Generator to yield on LM record at a time."""
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
        
all_lipids_file = 'LMSDFAll.sdf'

out_lm_file = 'lm_metadata.txt'

fields = ('CATEGORY', 'FORMULA', 'HMDBID', 'CHEBI_ID', 'KEGG_ID',
          'MAIN_CLASS', 'SUB_CLASS', 'SYSTEMATIC_NAME')

print ('processing {} ...'.format(all_lipids_file))
count = 0
with open(out_lm_file, 'w') as outfile:
    print('LM_ID\t'+'\t'.join(fields), file=outfile)
    
    for record in get_lm_records(all_lipids_file):
        metas = get_lm_metafields(record)
        print(metas['LM_ID'])
        line = [metas['LM_ID']]
        
        for field in fields:
            if field in metas:
                line.append(metas[field])
            else:
                line.append('')
        
        print('\t'.join(line), file=outfile)
        count +=1
    
print('{} records processed in {}'.format(count, all_lipids_file))
print('File {} generated'.format(out_lm_file))

# Extract information from HMDB

#expname = 'HMDB00010.xml'
all_mets_file = 'hmdb_metabolites.zip'
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
print('{} compounds in HMDB. {} had a KeGGId'.format(n_mets, n_trans))

out_filename = 'trans_hmdb2kegg.txt'

# try to follow KeGG format for Id translation.
line_frmt = 'hmdb:{}\tcpd:{}\tequivalent'.format

with open(out_filename, 'w') as outfile:
    for hmdb in sorted(list(conv.keys())):
        print(line_frmt(hmdb, conv[hmdb]), file=outfile)

print(out_filename, 'was created')

