from __future__ import print_function
import time
import os

import json
import requests

import pandas as pd

import tkinter as tk
from tkinter import filedialog

from metabolinks.masstrix import read_MassTRIX

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dbs'))

# ------------- Code related to Kegg compound records -------------------

def get_kegg_section(k_record, sname):
    """Get the section with name _sname_ from a kegg compound record.
    
       Returns a str, possibly empty."""
    
    in_section = False
    section = []
    
    for line in k_record.splitlines():
        if line.startswith(sname):
            in_section = True
            section.append(line)
        elif in_section and line.startswith(' '):
            section.append(line)
        elif in_section and not line.startswith(' '):
            break

    sectionlines = [line[12:] for line in section]
    return '\n'.join(sectionlines)


def load_local_kegg_db(db_fname):
    """Load Kegg compound records from a local text DB."""
    
    kegg = {}
    db = open(db_fname)
    curr_c = None
    curr_txt = None
    for line in db:
        if len(line.strip()) == 0:
            continue
        if line.startswith('---- Compound:'):
            if curr_c is not None:
                whole_rec = ''.join(curr_txt)
                kegg[curr_c] = whole_rec
            curr_c = line.split(':')[1].strip()
            curr_txt = []
        elif curr_c is not None:
            curr_txt.append(line)
    whole_rec = ''.join(curr_txt)
    kegg[curr_c] = whole_rec
    db.close()
    return kegg

def load_local_lipidmaps_db(db_fname):
    """Load LIPIDMAPS records from a local text DB."""
    df = pd.read_table(db_fname, index_col=0, dtype={'CHEBI_ID':str})
    return df

def get_dblinks_brite_sections(k_record):
    """Get sections DBLINKS and BRITE from a kegg compound record.
    
       Returns a dict."""
    
    dblinks = get_kegg_section(k_record, 'DBLINKS')
    brite = get_kegg_section(k_record, 'BRITE')
    return {'DBLINKS':dblinks, 'BRITE':brite}


def request_kegg_record(c_id, localdict=None):
    if localdict is not None:
        return localdict[c_id]
    return requests.get('http://rest.kegg.jp/get/' + c_id).text


# Helper. No used in example code (file names are hard coded).
def get_filename_using_tk():
    """Choose a filename using Tk"""
    root = tk.Tk()
    root.withdraw()
    fname = filedialog.askopenfilename(filetypes = [("TSV","*.tsv")])
    print ('Selected file {}'.format(fname))
    return fname

# Load ID translation tables as dicts
# IMPORTANT: Use local files, (fetched by fetch_dbs.py)
def get_trans_id_table(fname):
    d = {}
    with open(fname) as f:
        # hmdb:HMDB00002 \t cpd:C00986 \t equivalent
        for line in f:
            if len(line) == 0:
                continue
            foreign, cpd, equiv = line.split('\t')
            foreign = foreign.split(':')[1].strip()
            cpd = cpd.split(':')[1].strip()
            d[foreign] = cpd
    return d


class Progress(object):
    """A progress reporter of % done for a maximum (int) of self.total"""
    def __init__(self, total=1):
        self.reset(total)
    def tick(self):
        self.count +=1
        print(str(round((1-((self.total-self.count)/self.total))*100)) + "% done")
    def reset(self, total=1):
        self.total = total
        self.count = 0


def classes_from_brite(krecord, trace=False):
    """Parses classes from the BRITE section of a kegg record."""
    classes = [[], [], [], []]

    k = krecord.split('BRIT', 1)[1]
    k = k.split('DBLINKS', 1)[0]
    k = k.splitlines()
    l = []
    for x in k:
        if x.startswith('E       '):
            l.append('            ' + x[len('E       '):])
        else:
            l.append(x)
    l = "\n".join(l)

    l = l.split('           ')[1:]
    p = []
    for x in l:
        if not x.startswith('  '):
            x = ('??'+x)
            p.append(x)
        else:
            p.append(x)
    p = "".join(p)
    p = p.split('??')
    p = "\n".join(p)
    p = p.splitlines()
    for x in p:
        if x.startswith(' ') and not x.startswith('  '):
            classes[0].append(x[1:])
        elif x.startswith('  ') and not x.startswith('   '):
            classes[1].append(x[2:])
        elif x.startswith('   ') and not x.startswith('    '):
            classes[2].append(x[3:])
        elif x.startswith('    ') and not x.startswith('     '):
            classes[3].append(x[4:])

    classes = ['#'.join(c) for c in classes]
    if trace:
        print ('from BRITE:')
        print(tuple(classes))
    return tuple(classes)


def classes_from_lipidmaps(lm_id, trace=False):
    if trace:
        print('LIPIDMAPS id: {}'.format(lm_id))
    # print(lm_df.loc[lm_id])
##     f = requests.get('http://www.lipidmaps.org/rest/compound/lm_id/' + lm_id + '/all/json').text
##     s = json.loads(f)
##     if f == '[]':
##         return 'null', 'null', 'null', 'null'
##     mm = 'Lipids [LM]'
##     cc = s['core'] if s['core'] is not None else 'null'
##     ss = s['main_class'] if s['main_class'] is not None else 'null'
##     tt = s['sub_class'] if s['sub_class'] is not None else 'null'
    record = lm_df.loc[lm_id]
    cc = ['Lipids [LM]']
    for field in 'CATEGORY', 'MAIN_CLASS', 'SUB_CLASS':
        cc.append(record[field] if pd.notnull(record[field]) else 'null')
    a = tuple(cc)
    if trace:
        print(a)
    return a 


def annotate_compound(compound_id, c_counter, already_fetched, 
                      trace=False, kegg_db=None):
    c_id = None
    lm_id = None
    hmdb_id = None
    ks_id = None
    
    brite = ''
    dblinks = ''
    in_plants = ''
    
    trans_kegg = compound_id
    trans_lipidmaps = compound_id
    
    if compound_id.startswith('C'):
        c_id = compound_id
        if c_id in kegg2lipidmaps_dict:
            lm_id = hmdb2kegg_dict[c_id]
            trans_lipidmaps = lm_id
    
    elif compound_id.startswith('LM'):
        lm_id = compound_id
        if lm_id in lipidmaps2kegg_dict:
            c_id = lipidmaps2kegg_dict[lm_id]
            trans_kegg = c_id
    
    elif compound_id.startswith('HMDB'):
        hmdb_id = compound_id
        if hmdb_id in hmdb2kegg_dict:
            c_id = hmdb2kegg_dict[hmdb_id]
            trans_kegg = c_id
            if c_id in kegg2lipidmaps_dict:
                lm_id = kegg2lipidmaps_dict[c_id]
                trans_lipidmaps = lm_id

    else:
        pass # does MASSTrix use any other DBs for IDs?
    
    if trace:
        print('\n---- compound: {} ({}, {})'.format(compound_id, 
                                             trans_kegg, 
                                             trans_lipidmaps))
    
    if c_id in already_fetched or lm_id in already_fetched:
        if trace:
            print('already looked up')
        return (trans_kegg, trans_lipidmaps, '', '', '', '', '')
        
    if c_id is not None:
        krecord = request_kegg_record(c_id, localdict=kegg_db)
        if trace:
            print('(look up {})'.format(c_id))
        c_counter.inc_looks()
        already_fetched.append(c_id)
    
        brite = get_kegg_section(krecord, 'BRITE')
        dblinks = get_kegg_section(krecord, 'DBLINKS')
        
        if len(dblinks) > 0:
            # find if there are LIPIMAPS or KNApSAcK Xrefs in DBLINKS section
            if 'LIPIDMAPS:' in dblinks:
                for line in dblinks.splitlines():
                    if 'LIPIDMAPS:' in line:
                        lm_id = line.split(':')[1].strip()
                        trans_lipidmaps = lm_id
            if 'KNApSAcK' in dblinks:
                for line in dblinks.splitlines():
                    if 'KNApSAcK:' in line:
                        ks_id = line.split(':')[1].strip()
    
    # get compound classification hierarchy
    # first from BRITE section
    if (c_id is not None) and len(brite) > 0:
        classes = classes_from_brite(krecord, trace)
    
    # then from LIPIDMAPS
    elif (lm_id is not None):
        if trace:
            print('(look up {})'.format(lm_id))
        classes = classes_from_lipidmaps(lm_id, trace)
        c_counter.inc_looks()
        already_fetched.append(lm_id)

    else:
        classes = ('', '', '', '')
        if trace:
            print ('No BRITE, no LIPIDMAPS')
    
    # find if present in plants
    in_plants = ''
    if ks_id is not None:
        r = requests.post('http://kanaya.naist.jp/knapsack_jsp/information.jsp?sname=C_ID&word=' + ks_id)
        c_counter.inc_looks()
        if 'Plantae' in r.text:
            in_plants = 'Plantae'
            if trace:
                print(in_plants)
    
    result = [trans_kegg, trans_lipidmaps]
    result.extend(list(classes))
    result.append(in_plants)
    return tuple(result)

class _count_compounds(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.looks_count = 0
    def inc(self, n=1):
        self.count += n
    def inc_looks(self, n=1):
        self.looks_count += n
    def get_count(self):
        return self.count
    def get_looks_count(self):
        return self.looks_count


def annotate_all(peak, c_counter, trace=False, kegg_db=None):
    """Create Pandas Series with compound class annotations."""
    
    data = [[], [], [], [], [], [], []]
    
    if trace:
        print('\n++++++++ PEAK +++++++++++++')
    
    already_fetched = []
    
    for compound_id in peak.split('#'):
        c_counter.inc()
        
        d_compound = annotate_compound(compound_id, 
                                       c_counter, 
                                       already_fetched, 
                                       trace, kegg_db)
        
        for d, i in zip(data, d_compound):
            if len(i) > 0: # don't include empty strings
                d.append(i)
    
    progress.tick()
    
    compressed_data = []
    for i, d in enumerate(data):
        if i < 2:
            compressed_data.append(d)
        else:
            compressed_data.append(set(d))
    
    hash_data = ['#'.join(d) for d in compressed_data]    

    col_names = ['trans_KEGG_Ids',
                 'trans_LipidMaps_Ids',
                 'Major Class', 
                 'Class', 
                 'Secondary Class', 
                 'Tertiary Class',
                 'KNApSAcK']
    
    
    if trace:
        print('\nDATA:')
        for n, d in zip(col_names, hash_data):
            print('{:25s} : {}'.format(n, d))
    
    return pd.Series(hash_data, index=col_names)


def insert_taxonomy(df, trace=False, local_kegg_db=None):
    """Apply annotate_all to kegg or LIPIDMAPS ids."""
    
    progress.reset(total=len(df['raw_mass']))
    c_counter = _count_compounds()
    df = pd.concat([df, df['KEGG_cid'].apply(annotate_all, 
                                             args=(c_counter, 
                                             trace,
                                             local_kegg_db))], axis=1)
    print('\nDone! {} ids processed. {} DB lookups'.format(c_counter.get_count(), c_counter.get_looks_count()))
    return df


# Object to report progress of annotations.
progress = Progress()

hmdb2kegg_dict = get_trans_id_table(os.path.join(DB_DIR, 'trans_hmdb2kegg.txt'))
#lipidmaps2kegg_dict = get_trans_id_table('trans_lipidmaps2kegg.txt')
_lm_db_fname = (os.path.join(DB_DIR, 'lm_metadata.txt'))
print ('\nLoading LIPIDMAPS data from {}\n'.format(_lm_db_fname))
lm_df = load_local_lipidmaps_db(_lm_db_fname)    

lipidmaps2kegg_dict = lm_df['KEGG_ID'].dropna().to_dict()
# print(len(trans_table))
# for k, v in trans_table.items():
    # print(k, '->', v)

kegg2lipidmaps_dict = {}
for k, v in lipidmaps2kegg_dict.items():
    kegg2lipidmaps_dict[v] = k

if __name__ == '__main__':
        
    print ('\nStarting...\n')

    start_time = time.time()

    testfile_name = 'example_data/MassTRIX_output.tsv'

    results = read_MassTRIX(testfile_name)
    print ("File {} was read\n".format(testfile_name))

    # Clean up uniqueId and "isotope" cols
    results = results.cleanup_cols()

    # check result
    results.info()
    assert len(results.columns) == 12 # there are 24 - 12 = 12 columns
    assert len(results.index) == 15 # there are still 15 peaks
    #print(results.head(2))

    print ('\nStarting annotations...')

    # Use a local Kegg DB.
    kegg_db = load_local_kegg_db('dbs/kegg_db.txt')
    
    # Call the main driver function.
    results = insert_taxonomy(results, local_kegg_db=kegg_db, trace=True)

    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    print ("Finished in " + "%02dm%02ds" % (m, s))
    print('------------------------------------')

    # results
    results.info()
    # print(results.head())
    
    # Export the annotated dataframe into a MS-Excel file
    # Name it with the same name as the .tsv, replacing tail with '_raw.xlsx'
    
    out_fname = testfile_name[:-4]+'_raw.xlsx'
    results.to_excel(out_fname, header=True, index=False)
    print ("File {} written".format(out_fname))
